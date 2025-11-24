#include "Core.h"

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_tma_cycle(0),
      _num_systolic_array_per_core(config.num_systolic_array_per_core),
      _tma(id, config.dram_req_size) {
  _sram_size = _config.sram_size * 1024;
  _used_sram_size = 0;
  _sa_compute_pipeline.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_inst_count.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _stat_tot_skipped_inst.resize(static_cast<size_t>(Opcode::COUNT), 0);
  
  // Initialize SRAM bandwidth model
  _sram_bw_model_enabled = config.sram_model_enabled;
  _sram_bytes_per_cycle = config.sram_bytes_per_cycle;
  _sram_available_tokens = _sram_bytes_per_cycle;
  
  // Initialize SRAM bandwidth statistics
  _stat_sram_total_bytes_read = 0;
  _stat_sram_total_bytes_written = 0;
  _stat_tot_sram_total_bytes_read = 0;
  _stat_tot_sram_total_bytes_written = 0;
  
  if (_sram_bw_model_enabled) {
    spdlog::info("[Config/Core {}] SRAM BW Model: Enabled, {} B/cycle", 
                 _id, _sram_bytes_per_cycle);
  }
}

bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  /* Check SRAM is enough to run tile */
  return _tiles.size() < 4  && !op->is_stonne_tile();
}

void Core::issue(std::shared_ptr<Tile> op) {
  if (op->get_instructions().size()){
    spdlog::trace("[Core {}][{}] New Tile is issued, remain sram: {} Required size: {}, Free size: {}",
      _id, _core_cycle, _sram_size-_used_sram_size, op->get_required_sram_size(),
      op->get_instructions().back()->get_free_sram_size());
  } else {
    spdlog::trace("[Core {}][{}] New Tile is issued, remain sram: {} Required size: {}",
      _id, _core_cycle, _sram_size-_used_sram_size, op->get_required_sram_size());
  }
  //_used_sram_size += op->get_required_sram_size();
  for (const auto& inst : op->get_instructions()) {
    if (inst->is_ready())
      op->enqueue_ready(inst);
  }
  _tiles.push_back(std::move(op));
}

std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

std::queue<std::shared_ptr<Instruction>>& Core::get_compute_pipeline(int compute_type) {
  if (compute_type == VECTOR_UNIT)
    return _vu_compute_pipeline;
  else if (compute_type == MATMUL || compute_type == PRELOAD) {
    uint32_t sa_idx = _systolic_array_rr;
    _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
    return _sa_compute_pipeline.at(sa_idx);
  }
  else {
    spdlog::error("Undefined compute type");
    exit(EXIT_FAILURE);
  }
}

void Core::vu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_vu_compute_pipeline.empty()) {
      if(_vu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        int bubble = _vu_compute_pipeline.front()->bubble_cycle;
        _stat_vu_compute_idle_cycle += bubble;
        _stat_vu_compute_cycle -= bubble;
        finish_instruction(_vu_compute_pipeline.front());
        _vu_compute_pipeline.pop();
      } else {
        // Charge SRAM bandwidth ONLY during active compute cycles
        auto& active_inst = _vu_compute_pipeline.front();
        size_t per_cycle_traffic = calculate_sram_traffic(active_inst);
        if (per_cycle_traffic > 0) {
          uint64_t read_bytes = (per_cycle_traffic * 2) / 3;
          uint64_t write_bytes = per_cycle_traffic / 3;
          _stat_sram_total_bytes_read += read_bytes;
          _stat_sram_total_bytes_written += write_bytes;
        }
        
        _stat_vu_compute_cycle++;
        retry = false;
      }
    } else {
      _stat_vu_compute_idle_cycle++;
      retry = false;
    }
  }
}

void Core::sa_cycle() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    bool retry = true;
    while (retry) {
      if (!_sa_compute_pipeline.at(i).empty()) {
        auto& active_inst = _sa_compute_pipeline.at(i).front();
        
        if(_sa_compute_pipeline.at(i).front()->finish_cycle <= _core_cycle) {
          int bubble = _sa_compute_pipeline.at(i).front()->bubble_cycle;
          _stat_sa_compute_idle_cycle.at(i) += bubble;
          _stat_sa_compute_cycle.at(i) -= bubble;
          finish_instruction(_sa_compute_pipeline.at(i).front());
          _sa_compute_pipeline.at(i).pop();
        } else {
          // Charge SRAM bandwidth ONLY during active compute cycles
          // Only charge when we're incrementing the compute cycle counter
          size_t per_cycle_traffic = calculate_sram_traffic(active_inst);
          if (per_cycle_traffic > 0) {
            uint64_t read_bytes = (per_cycle_traffic * 2) / 3;
            uint64_t write_bytes = per_cycle_traffic / 3;
            _stat_sram_total_bytes_read += read_bytes;
            _stat_sram_total_bytes_written += write_bytes;
          }
          
          _stat_sa_compute_cycle.at(i)++;
          retry = false;
        }
      } else {
        _stat_sa_compute_idle_cycle.at(i)++;
        retry = false;
      }
    }
  }
}

void Core::compute_cycle() {
  vu_cycle();
  sa_cycle();
}

void Core::dma_cycle() {
  /* Check finished dma operation */
  while(_dma_finished_queue.size()) {
    std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
    assert(instruction->get_waiting_request()==0);

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      auto& key = instruction->get_tag_id();
      assert(!_tma.get_tag_finish(instruction->subgraph_id, key));
      _tma.set_tag_finish(instruction->subgraph_id, key);
      spdlog::trace("[Core {}][{}] {} ASYNC FINISHED, Used sram: {}, Release sram: {}, subgraph_id: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}",
                    _id, _core_cycle, opcode_to_string(instruction->get_opcode()),
                    _used_sram_size, instruction->get_free_sram_size(),
                    instruction->subgraph_id, instruction->get_addr_name(),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_id(), ", ")),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_idx_list(), ", ")),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_stride_list(), ", ")));
      for (auto & wait_inst : _tma.get_tag_waiter(instruction->subgraph_id, key)) {
        _tma.mark_tag_used(instruction->subgraph_id, key);
        finish_instruction(wait_inst);
      }
    }
    _dma_finished_queue.erase(_dma_finished_queue.begin());
  }

  if (_tma.is_finished()) {
    /* Finish instruction when it is DMA store */
    if (_tma.get_current_inst() != nullptr) {
      std::shared_ptr<Instruction> finished_inst = std::move(_tma.get_current_inst());
      if (finished_inst->is_dma_write()) {
        /* Only DMA write operation is finished! */
        finish_instruction(finished_inst);
      } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
        /* Register tag table for async dma load */
        _tma.register_tag(finished_inst->subgraph_id, finished_inst->get_tag_id());
        finish_instruction(finished_inst);
      } else if(!finished_inst->is_dma_read()) {
        spdlog::error("[Core {}][{}] TMA instruction in not valid", _id, _core_cycle);
        exit(EXIT_FAILURE);
      } else if (finished_inst->get_opcode() == Opcode::BAR) {
        spdlog::trace("[Core {}][{}] {} FINISHED, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _id, _core_cycle,
                      opcode_to_string(finished_inst->get_opcode()), finished_inst->get_addr_name(),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_id(), ", ")),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_idx_list(), ", ")),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_stride_list(), ", ")));
      }
      /*Pass to waiting queue */
      _dma_waiting_queue[finished_inst.get()] = std::move(finished_inst);
    }

    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _ld_inst_queue.front();
      _tma.issue_tile(inst);
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _st_inst_queue.front();
      _tma.issue_tile(inst);
      _st_inst_queue.pop();
    } else {
      /* TMA is idle */
      _stat_tma_idle_cycle++;
      return;
    }
  }
  /* Generate memfetch */
  auto access_vec = _tma.get_memory_access();
  for (auto access : *access_vec) {
    access->set_start_cycle(_core_cycle);
    _request_queue.push(access);
  }

  /* Increase tma stat cycle */
  _stat_tma_cycle++;
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();
  
  /* Refill SRAM bandwidth tokens at beginning of cycle */
  refill_sram_bandwidth();

  /* Increase core cycle counter */
  _core_cycle++;

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    auto& instructions = _tiles[i]->get_ready_instructions();
    for (auto it=instructions.begin(); it!=instructions.end();) {
      auto& inst = *it;
      /* Skip instruction is not ready  */
      //if (!inst->is_ready())
      //  continue;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          {
            /* Check another MOVIN with same tag is issued */
            auto& key = inst->get_tag_id();
            if (inst->is_sparse_inst()) {
              _tma.register_tag(inst->subgraph_id, key);
              _tma.set_tag_sparse(inst->subgraph_id, key);
              finish_instruction(inst);
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else if (inst->is_async_dma() && _tma.tag_key_exist(inst->subgraph_id, key)) {
              bool finished = _tma.get_tag_finish(inst->subgraph_id, key);
              if (finished)
                finish_instruction(inst);
              else
                _tma.register_tag_waiter(inst->subgraph_id, key, inst);
              spdlog::trace("[Core {}][{}] {} SKIPPED, free_sram_size: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _id, _core_cycle,
                            opcode_to_string(inst->get_opcode()), inst->get_free_sram_size(),
                            inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else {
              spdlog::trace("[Core {}][{}] {} ISSUED, free_sram_size: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _id, _core_cycle,
                            opcode_to_string(inst->get_opcode()), inst->get_free_sram_size(),
                            inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
              _ld_inst_queue.push(inst);
              issued = true;
              break;
            }
          }
        case Opcode::MOVOUT:
          spdlog::trace("[Core {}][{}] {} ISSUED, free_sram_size: {}", _id, _core_cycle,
                        opcode_to_string(inst->get_opcode()), inst->get_free_sram_size());
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            auto& target_pipeline = get_compute_pipeline(inst->get_compute_type());
            if (target_pipeline.empty()) {
              inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
              inst->bubble_cycle = inst->get_overlapping_cycle();
            } else {
              int overlapped_cycle = std::min(target_pipeline.back()->finish_cycle - _core_cycle, inst->get_overlapping_cycle());
              int bubble_cycle = inst->get_overlapping_cycle() - overlapped_cycle;
              inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle() - overlapped_cycle;
              inst->bubble_cycle = bubble_cycle;
            }
            if (inst->get_compute_cycle() == 0) {
              inst->finish_instruction();
              static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              instructions.erase(it);
            } else {
              // SRAM bandwidth is now tracked during execution in sa_cycle()/vu_cycle()
              spdlog::trace("[Core {}][SA {}][{}] {}-{} ISSUED, finsh at {}", _id, _systolic_array_rr, _core_cycle,
                            opcode_to_string(inst->get_opcode()), inst->get_compute_type(), inst->finish_cycle);
              target_pipeline.push(inst);
              issued = true;
              if (inst->get_compute_type()) {
                _stat_gemm_inst++;
              }
            }
          }
          break;
        case Opcode::BAR:
          {
            auto& key = inst->get_tag_id();
            uint32_t finished = _tma.get_tag_finish(inst->subgraph_id, key);
            if (finished == -1) {
              for (auto child_inst : inst->get_child_inst()) {
                if (child_inst->get_opcode() == Opcode::COMP && child_inst->get_compute_type() == MATMUL) {
                  child_inst->set_compute_cycle(0);
                }
              }
              finish_instruction(inst);
            } else if (finished != 0) {
              _tma.mark_tag_used(inst->subgraph_id, key);
              finish_instruction(inst);
            } else {
              _tma.register_tag_waiter(inst->subgraph_id, key, inst);
            }
            spdlog::trace("[Core {}][{}] {} ISSUED,  addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _id, _core_cycle,
                            opcode_to_string(inst->get_opcode()), inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
            issued = true;
          }
          break;
        default:
          spdlog::error("Undefined instruction opcode type");
          exit(EXIT_FAILURE);
      }

      if (issued) {
        _stat_inst_count.at(static_cast<size_t>(inst->get_opcode()))++;
        instructions.erase(it);
        break;
      }
      it++;
    }
  }

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

void Core::finish_instruction(std::shared_ptr<Instruction>& inst) {
  size_t free_sram_size = inst->get_free_sram_size();
  if (inst->finished) {
    spdlog::error("[Core {}][{}] {} FINISHED, inst already finished!!", _id, _core_cycle,
                  opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  if (inst->get_opcode() == Opcode::COMP) {
    spdlog::trace("[Core {}][{}] {}-{} FINISHED, Used sram: {}, Release sram: {}",
      _id, _core_cycle, opcode_to_string(inst->get_opcode()), inst->get_compute_type(),
      _used_sram_size, inst->get_free_sram_size());
  } else if (inst->get_opcode() != Opcode::BAR && inst->is_async_dma()){
    spdlog::trace("[Core {}][{}] {} ASYNC REGISTERED, Used sram: {}, Release sram: {} subgraph_id: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}",
      _id, _core_cycle, opcode_to_string(inst->get_opcode()), _used_sram_size,
      inst->get_free_sram_size(), inst->subgraph_id, inst->get_addr_name(),
      inst->get_tag_id(),
      fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
      fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
  } else if ((inst->get_opcode() == Opcode::MOVIN || inst->get_opcode() == Opcode::MOVOUT) && !inst->is_async_dma()) {
    spdlog::trace("[Core {}][{}] {} FINISHED, free_sram_size: {} addr_name: {}", _id, _core_cycle,
      opcode_to_string(inst->get_opcode()), inst->get_free_sram_size(),
      inst->get_addr_name());
  }
  //_used_sram_size -= free_sram_size;
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_vu_compute_pipeline.empty();
  for (int i=0; i<_num_systolic_array_per_core;i++)
    running = running || !_sa_compute_pipeline.at(i).empty();
  running = running || !_dma_waiting_queue.empty() || !_dma_finished_queue.empty();
  running = running || !_tma.empty();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() {
  return !_request_queue.empty();
}

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(mem_fetch* response) {
  Instruction* owner_inst = static_cast<Instruction*>(response->get_custom_data());
  assert(owner_inst->get_waiting_request());

  owner_inst->dec_waiting_request();
  if (!owner_inst->get_waiting_request()) {
    auto it = _dma_waiting_queue.find(owner_inst);
    if (it != _dma_waiting_queue.end()) {
      std::shared_ptr<Instruction> moved_inst = std::move(it->second);
      _dma_finished_queue.push_back(std::move(moved_inst));
      _dma_waiting_queue.erase(it);
    } else {
      assert(true || "Can't happend...!");
    }
  }
  _stat_mem_response++;
  delete response;
}

bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  std::vector<float> sa_utilization;
  update_stats();
  spdlog::info("===== Instructions count =====");
  for (int i=0; i < static_cast<size_t>(Opcode::COUNT); i++) {
    if (i == static_cast<size_t>(Opcode::COMP))
      spdlog::info("Core [{}] : {} inst count {} (GEMM: {}, Vector: {}), skipped inst count {}", _id, opcode_to_string(static_cast<Opcode>(i)), _stat_inst_count.at(i), _stat_gemm_inst, _stat_inst_count.at(i) - _stat_gemm_inst, _stat_tot_skipped_inst.at(i));
    else
      spdlog::info("Core [{}] : {} inst count {}, skipped inst count {}", _id, opcode_to_string(static_cast<Opcode>(i)), _stat_inst_count.at(i), _stat_tot_skipped_inst.at(i));
  }
  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_tot_sa_compute_cycle.at(i) * 100) / _core_cycle);
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] Utilization(%) {:.2f}, active cycle {}, idle cycle {}", _id, i, sa_utilization.at(i),
      _stat_tot_sa_compute_cycle.at(i), _stat_tot_sa_compute_idle_cycle.at(i));
  float dram_bw = _config.dram_req_size * _stat_tot_mem_response * _config.core_freq / (_core_cycle * 1000); // B/cycle
  spdlog::info("Core [{}] : TMA active cycle {} TMA idle cycle {} DRAM BW {:.3f} GB/s ({})", _id, _stat_tot_tma_cycle, _stat_tot_tma_idle_cycle, dram_bw, _stat_tot_mem_response);
  spdlog::info("Core [{}] : Vector Unit Utilization(%) {:.2f}, active cycle {}, idle_cycle {}", _id,
    static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle, _stat_tot_vu_compute_idle_cycle);
  
  // Print SRAM bandwidth statistics if enabled
  if (_sram_bw_model_enabled) {
    uint64_t total_sram_bytes = _stat_tot_sram_total_bytes_read + _stat_tot_sram_total_bytes_written;
    double sram_bw_gb_s = total_sram_bytes * _config.core_freq / (_core_cycle * 1000.0); // GB/s
    double sram_utilization = (sram_bw_gb_s / (_sram_bytes_per_cycle * _config.core_freq / 1000.0)) * 100.0;
    
    spdlog::info("======= SRAM Bandwidth =======");
    spdlog::info("Core [{}] : SRAM Total Read {} MB, Total Write {} MB", 
                 _id, _stat_tot_sram_total_bytes_read/(1024*1024),
                 _stat_tot_sram_total_bytes_written/(1024*1024));
    spdlog::info("Core [{}] : SRAM BW Usage {:.2f} GB/s, Utilization {:.1f}%",
                 _id, sram_bw_gb_s, sram_utilization);
  }
  
  spdlog::info("Core [{}] : Numa hit count : {}, Numa miss count : {}", _id, _stat_numa_hit, _stat_numa_miss);
  spdlog::info("Core [{}] : Total cycle {}", _id, _core_cycle);
}

void Core::print_current_stats() {
  std::vector<float> sa_utilization;
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_sa_compute_cycle.at(i) * 100) / _config.core_print_interval);
  float dram_bw = _config.dram_req_size * _stat_mem_response * _config.core_freq / (_config.core_print_interval * 1000); // B/cycle
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;

  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] Utilization(%) {:.2f}, active cycle {}, idle cycle {}", _id, i, sa_utilization.at(i),
      _stat_sa_compute_cycle.at(i), _stat_sa_compute_idle_cycle.at(i));
  spdlog::info("Core [{}] : TMA active cycle {} TMA idle cycle {} DRAM BW {:.3f} GB/s ({})", _id, _stat_tma_cycle, _stat_tma_idle_cycle, dram_bw, _stat_mem_response);
  spdlog::info("Core [{}] : Vector Unit Utilization(%) {:.2f}, active cycle {}, idle_cycle {}", _id,
    static_cast<float>(_stat_vu_compute_cycle * 100) / _config.core_print_interval, _stat_vu_compute_cycle, _stat_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : Total cycle {}", _id, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    _stat_tot_sa_compute_cycle.at(i) += _stat_sa_compute_cycle.at(i);
    _stat_tot_sa_compute_idle_cycle.at(i) += _stat_sa_compute_idle_cycle.at(i);
    _stat_sa_compute_cycle.at(i) = 0;
    _stat_sa_compute_idle_cycle.at(i) = 0;
  }

  _stat_tot_vu_compute_cycle += _stat_vu_compute_cycle;
  _stat_tot_tma_cycle += _stat_tma_cycle;
  _stat_tot_tma_idle_cycle += _stat_tma_idle_cycle;
  _stat_tot_mem_response += +_stat_mem_response;

  _stat_vu_compute_cycle = 0;
  _stat_tma_cycle = 0;
  _stat_tma_idle_cycle = 0;
  _stat_vu_compute_idle_cycle = 0;
  _stat_mem_response = 0;
  
  // Update SRAM bandwidth stats
  _stat_tot_sram_total_bytes_read += _stat_sram_total_bytes_read;
  _stat_tot_sram_total_bytes_written += _stat_sram_total_bytes_written;
  _stat_sram_total_bytes_read = 0;
  _stat_sram_total_bytes_written = 0;
}

size_t Core::calculate_sram_traffic(std::shared_ptr<Instruction>& inst) {
  if (!_sram_bw_model_enabled || inst->get_opcode() != Opcode::COMP) {
    return 0;
  }
  
  // Get tile information
  size_t tile_bytes = inst->get_tile_numel() * inst->get_precision();
  
  // Option 1: Tile-based calculation (when tile size is available)
  if (tile_bytes > 0 && inst->get_compute_cycle() > 0) {
    // Calculate SRAM traffic based on compute type
    double traffic_factor;
    
    if (inst->get_compute_type() == MATMUL) {
      // For tiled GEMM with systolic array data reuse:
      // Assuming 128×128×128 tiles on 128×128 SA:
      // - Read A: 64 KB (128×128×4)
      // - Read B: 64 KB (128×128×4) 
      // - Read partial C: 64 KB (for accumulation)
      // - Write C: 64 KB
      // Total: 256 KB over ~128 cycles = 2048 B/cycle
      // 
      // But with weight reuse and pipelining, effective traffic is lower
      // Traffic factor accounts for: A read + B read + C read/write
      // With data reuse: ~2.0× tile size is more realistic
      traffic_factor = 2.0;
    } else {
      // Element-wise operations: 2 reads + 1 write
      traffic_factor = 3.0;
    }
    
    size_t total_traffic = static_cast<size_t>(tile_bytes * traffic_factor);
    size_t per_cycle_traffic = total_traffic / inst->get_compute_cycle();
    
    spdlog::trace("[Core {}][{}] SRAM traffic (tile-based): {} B total over {} cycles = {} B/cycle (factor={})", 
                  _id, _core_cycle, total_traffic, inst->get_compute_cycle(), per_cycle_traffic, traffic_factor);
    
    return per_cycle_traffic;
  }
  
  // Option 2: Architecture-based estimate (when tile size not available)
  // Based on 128×128 systolic array with 128×128×128 tile size
  if (inst->get_compute_cycle() > 0) {
    size_t bytes_per_cycle;
    
    if (inst->get_compute_type() == MATMUL) {
      // For 128×128×128 tile on 128×128 SA:
      // - Total traffic: ~256 KB (A + B + partial C + C write)
      // - Compute cycles: ~128 cycles
      // - Theoretical: 256 KB / 128 = 2048 B/cycle
      // 
      // However, with weight stationary dataflow and data reuse:
      // - Weights loaded once and reused (amortized)
      // - Inputs streamed: 128 elements/cycle × 4 bytes = 512 B/cycle
      // - Outputs accumulated in registers, written once (amortized)
      // 
      // Conservative estimate accounting for all traffic: 1024 B/cycle
      bytes_per_cycle = 1024;
      
      spdlog::trace("[Core {}][{}] SRAM traffic (MATMUL estimate): {} B/cycle", 
                    _id, _core_cycle, bytes_per_cycle);
    } else {
      // Vector operations have less data reuse
      // Assume 2 reads + 1 write per element processed
      // For vector unit processing multiple elements/cycle: ~512 B/cycle
      bytes_per_cycle = 512;
      
      spdlog::trace("[Core {}][{}] SRAM traffic (Vector estimate): {} B/cycle", 
                    _id, _core_cycle, bytes_per_cycle);
    }
    
    return bytes_per_cycle;
  }
  
  return 0;
}

void Core::charge_sram_bandwidth(std::shared_ptr<Instruction>& inst) {
  if (!_sram_bw_model_enabled || inst->get_opcode() != Opcode::COMP) {
    return;
  }
  
  size_t required_bytes = calculate_sram_traffic(inst);
  
  // For unified bandwidth model (Phase 1), we just track total bytes
  // Split read/write for statistics only (assuming 2 reads : 1 write ratio)
  uint64_t read_bytes = (required_bytes * 2) / 3;   // 2/3 of traffic
  uint64_t write_bytes = required_bytes / 3;         // 1/3 of traffic
  
  // Charge unified bandwidth pool
  _sram_available_tokens -= required_bytes;
  
  // Update statistics
  _stat_sram_total_bytes_read += read_bytes;
  _stat_sram_total_bytes_written += write_bytes;
  
  spdlog::trace("[Core {}][{}] SRAM BW charged: {} B total ({} R + {} W), available: {:.0f} B", 
                _id, _core_cycle, required_bytes, read_bytes, write_bytes, _sram_available_tokens);
}

void Core::refill_sram_bandwidth() {
  if (!_sram_bw_model_enabled) {
    return;
  }
  
  // Refill bandwidth tokens at the beginning of each cycle
  _sram_available_tokens = std::min(_sram_available_tokens + _sram_bytes_per_cycle,
                                     _sram_bytes_per_cycle);
}
import torch
import torch._dynamo
import torch.utils.cpp_extension
import sys
import argparse

def test_result(name, out, cpu_out, rtol=1e-4, atol=1e-4):
    if torch.allclose(out.cpu(), cpu_out, rtol=rtol, atol=atol):
        message = f"|{name} Test Passed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
    else:
        message = f"|{name} Test Failed|"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        print("custom out: ", out.cpu())
        print("cpu out: ", cpu_out)
        exit(1)

def test_bGEMM(device, batch_size=1, m=32, n=16, k=64):
    """
    Batched GEMM test function
    Args:
        device: Target device for computation
        batch_size: Number of matrices in the batch
        m: Number of rows in matrix A
        n: Number of columns in matrix B
        k: Shared dimension (columns of A, rows of B)
    
    Matrix dimensions:
        A: [batch_size, m, k]
        B: [batch_size, n, k] 
        Result: [batch_size, m, n]
    """
    def bmm(a, b):
        return torch.bmm(a, b.transpose(1, 2))
    
    torch.manual_seed(0)
    a = torch.randn(batch_size, m, k).to(device=device)
    b = torch.randn(batch_size, n, k).to(device=device)
    opt_fn = torch.compile(dynamic=False)(bmm)
    res = opt_fn(a, b)
    out = bmm(a.cpu(), b.cpu())
    
    print(f"Testing bGEMM: batch={batch_size}, M={m}, N={n}, K={k}")
    print(f"  Input A shape: [{batch_size}, {m}, {k}]")
    print(f"  Input B shape: [{batch_size}, {n}, {k}]")
    print(f"  Output shape:  [{batch_size}, {m}, {n}]")
    test_result("Batched GEMM", res, out)

if __name__ == "__main__":
    import os
    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/workspace/PyTorchSim'))

    from Scheduler.scheduler import ExecutionEngine
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Batched GEMM test with specified dimensions')
    parser.add_argument('--batch', type=int, default=None, help='Batch size (B)')
    parser.add_argument('--m', type=int, default=None, help='M dimension (rows of A)')
    parser.add_argument('--k', type=int, default=None, help='K dimension (cols of A, rows of B)')
    parser.add_argument('--n', type=int, default=None, help='N dimension (cols of B)')
    args = parser.parse_args()
    
    module = ExecutionEngine.setup_device()
    device = module.custom_device()
    
    print("=" * 60)
    print("Running Batched GEMM Tests")
    print("=" * 60)
    
    # If dimensions are specified via command line, run single test
    if args.batch is not None and args.m is not None and args.k is not None and args.n is not None:
        print(f"\nRunning custom test: B={args.batch}, M={args.m}, K={args.k}, N={args.n}")
        test_bGEMM(device, batch_size=args.batch, m=args.m, n=args.n, k=args.k)
    else:
        # Run default test suite
        print("\nRunning default test suite...")
        test_bGEMM(device, batch_size=1, m=32, n=32, k=32)
        test_bGEMM(device, batch_size=2, m=256, n=128, k=256)
        test_bGEMM(device, batch_size=2, m=128, n=256, k=256)
        test_bGEMM(device, batch_size=2, m=256, n=256, k=128)
        test_bGEMM(device, batch_size=4, m=256, n=256, k=256)
        test_bGEMM(device, batch_size=12, m=512, n=512, k=64)
        test_bGEMM(device, batch_size=16, m=512, n=512, k=64)
    
    print("=" * 60)
    print("All Batched GEMM Tests Completed Successfully!")
    print("=" * 60)

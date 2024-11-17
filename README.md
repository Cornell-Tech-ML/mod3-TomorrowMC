# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


## NUMBA diagnostics:
```python
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (155)
  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (155) 
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        aligned = (                                                      | 
                len(out_strides) == len(in_strides)                      | 
                and (out_strides == in_strides).all()--------------------| #0
                and (out_shape == in_shape).all()------------------------| #1
        )                                                                | 
                                                                         | 
        if aligned:                                                      | 
            for elem_idx in prange(len(out)):----------------------------| #2
                out[elem_idx] = fn(in_storage[elem_idx])                 | 
        else:                                                            | 
            for elem_idx in prange(len(out)):----------------------------| #3
                out_pos = np.empty(MAX_DIMS, np.int32)                   | 
                in_pos = np.empty(MAX_DIMS, np.int32)                    | 
                to_index(elem_idx, out_shape, out_pos)                   | 
                broadcast_index(out_pos, out_shape, in_shape, in_pos)    | 
                out[index_to_position(out_pos, out_strides)] = fn(       | 
                    in_storage[index_to_position(in_pos, in_strides)]    | 
                )                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (174) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_pos = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (175) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_pos = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (209)
  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (209) 
--------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                         | 
        out: Storage,                                                                 | 
        out_shape: Shape,                                                             | 
        out_strides: Strides,                                                         | 
        a_storage: Storage,                                                           | 
        a_shape: Shape,                                                               | 
        a_strides: Strides,                                                           | 
        b_storage: Storage,                                                           | 
        b_shape: Shape,                                                               | 
        b_strides: Strides,                                                           | 
    ) -> None:                                                                        | 
        aligned_strides = (                                                           | 
                len(out_strides) == len(a_strides) == len(b_strides)                  | 
                and (out_strides == a_strides).all()----------------------------------| #4
                and (out_strides == b_strides).all()----------------------------------| #5
                and (out_shape == a_shape).all()--------------------------------------| #6
                and (out_shape == b_shape).all()--------------------------------------| #7
        )                                                                             | 
                                                                                      | 
        if aligned_strides:                                                           | 
            for pos in prange(len(out)):----------------------------------------------| #8
                out[pos] = fn(a_storage[pos], b_storage[pos])                         | 
        else:                                                                         | 
            for pos in prange(len(out)):----------------------------------------------| #9
                out_coords = np.empty(MAX_DIMS, np.int32)                             | 
                a_coords = np.empty(MAX_DIMS, np.int32)                               | 
                b_coords = np.empty(MAX_DIMS, np.int32)                               | 
                to_index(pos, out_shape, out_coords)                                  | 
                broadcast_index(out_coords, out_shape, a_shape, a_coords)             | 
                broadcast_index(out_coords, out_shape, b_shape, b_coords)             | 
                a_val = a_storage[index_to_position(a_coords, a_strides)]             | 
                b_val = b_storage[index_to_position(b_coords, b_strides)]             | 
                out[index_to_position(out_coords, out_strides)] = fn(a_val, b_val)    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (233) 
is hoisted out of the parallel loop labelled #9 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_coords = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (234) 
is hoisted out of the parallel loop labelled #9 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_coords = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (235) 
is hoisted out of the parallel loop labelled #9 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_coords = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (267)
  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (267) 
------------------------------------------------------------------|loop #ID
    def _reduce(                                                  | 
        out: Storage,                                             | 
        out_shape: Shape,                                         | 
        out_strides: Strides,                                     | 
        a_storage: Storage,                                       | 
        a_shape: Shape,                                           | 
        a_strides: Strides,                                       | 
        reduce_dim: int,                                          | 
    ) -> None:                                                    | 
        reduction_size = a_shape[reduce_dim]                      | 
        reduction_stride = a_strides[reduce_dim]                  | 
                                                                  | 
        for base_idx in prange(len(out)):-------------------------| #10
            position = np.empty(MAX_DIMS, np.int32)               | 
            to_index(base_idx, out_shape, position)               | 
            out_pos = index_to_position(position, out_strides)    | 
            current = out[out_pos]                                | 
            a_pos = index_to_position(position, a_strides)        | 
                                                                  | 
            for offset in range(reduction_size):                  | 
                current = fn(current, a_storage[a_pos])           | 
                a_pos += reduction_stride                         | 
                                                                  | 
            out[out_pos] = current                                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (280) 
is hoisted out of the parallel loop labelled #10 (it will be performed before 
the loop is executed and reused inside the loop):
   Allocation:: position = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (295)
  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/yifei.hu/NoUpdate/Python/MLE/mod3-TomorrowMC/minitorch/fast_ops.py (295) 
------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                          | 
    out: Storage,                                                                                     | 
    out_shape: Shape,                                                                                 | 
    out_strides: Strides,                                                                             | 
    a_storage: Storage,                                                                               | 
    a_shape: Shape,                                                                                   | 
    a_strides: Strides,                                                                               | 
    b_storage: Storage,                                                                               | 
    b_shape: Shape,                                                                                   | 
    b_strides: Strides,                                                                               | 
) -> None:                                                                                            | 
    """NUMBA tensor matrix multiply function.                                                         | 
                                                                                                      | 
    Should work for any tensor shapes that broadcast as long as                                       | 
                                                                                                      | 
    ```                                                                                               | 
    assert a_shape[-1] == b_shape[-2]                                                                 | 
    ```                                                                                               | 
                                                                                                      | 
    Optimizations:                                                                                    | 
                                                                                                      | 
    * Outer loop in parallel                                                                          | 
    * No index buffers or function calls                                                              | 
    * Inner loop should have no global writes, 1 multiply.                                            | 
                                                                                                      | 
                                                                                                      | 
    Args:                                                                                             | 
    ----                                                                                              | 
        out (Storage): storage for `out` tensor                                                       | 
        out_shape (Shape): shape for `out` tensor                                                     | 
        out_strides (Strides): strides for `out` tensor                                               | 
        a_storage (Storage): storage for `a` tensor                                                   | 
        a_shape (Shape): shape for `a` tensor                                                         | 
        a_strides (Strides): strides for `a` tensor                                                   | 
        b_storage (Storage): storage for `b` tensor                                                   | 
        b_shape (Shape): shape for `b` tensor                                                         | 
        b_strides (Strides): strides for `b` tensor                                                   | 
                                                                                                      | 
    Returns:                                                                                          | 
    -------                                                                                           | 
        None : Fills in `out`                                                                         | 
                                                                                                      | 
    """                                                                                               | 
    #TODO                                                                                             | 
    batch_stride_a = a_strides[0] if a_shape[0] > 1 else 0                                            | 
    batch_stride_b = b_strides[0] if b_shape[0] > 1 else 0                                            | 
                                                                                                      | 
    for batch in prange(out_shape[0]):----------------------------------------------------------------| #13
        batch_offset_a = batch * batch_stride_a                                                       | 
        batch_offset_b = batch * batch_stride_b                                                       | 
                                                                                                      | 
        for row in prange(out_shape[1]):--------------------------------------------------------------| #12
            row_offset_a = batch_offset_a + row * a_strides[1]                                        | 
                                                                                                      | 
            for col in prange(out_shape[2]):----------------------------------------------------------| #11
                col_offset_b = batch_offset_b + col * b_strides[2]                                    | 
                result = 0.0                                                                          | 
                pos_a = row_offset_a                                                                  | 
                pos_b = col_offset_b                                                                  | 
                                                                                                      | 
                for _ in range(a_shape[2]):                                                           | 
                    result += a_storage[pos_a] * b_storage[pos_b]                                     | 
                    pos_a += a_strides[2]                                                             | 
                    pos_b += b_strides[1]                                                             | 
                                                                                                      | 
                out[batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]] = result    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

```
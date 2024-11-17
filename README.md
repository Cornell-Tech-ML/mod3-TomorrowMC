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


## Task 3.1/3.2 NUMBA diagnostics:
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

## Task 3.5
```markdown
==================================================
Training on Simple Dataset
==================================================
Epoch  0  time  6.782475709915161
Epoch   0 | Loss:     7.4889 | Accuracy:  86.00% | Correct:  43/50
Epoch  1  time  0.3869459629058838
Epoch  2  time  0.37249207496643066
Epoch  3  time  0.37729716300964355
Epoch  4  time  0.38715291023254395
Epoch  5  time  0.3830997943878174
Epoch  6  time  0.39457225799560547
Epoch  7  time  0.37479400634765625
Epoch  8  time  0.37906885147094727
Epoch  9  time  0.3829360008239746
Epoch  10  time  0.3778400421142578
Epoch  10 | Loss:     1.9843 | Accuracy:  90.00% | Correct:  45/50
Epoch  11  time  0.4018287658691406
Epoch  12  time  0.38393306732177734
Epoch  13  time  0.3763282299041748
Epoch  14  time  0.3704400062561035
Epoch  15  time  0.3805058002471924
Epoch  16  time  0.41182994842529297
Epoch  17  time  0.3959991931915283
Epoch  18  time  0.37145423889160156
Epoch  19  time  0.3753819465637207
Epoch  20  time  0.38216233253479004
Epoch  20 | Loss:     1.6707 | Accuracy:  98.00% | Correct:  49/50
Epoch  21  time  0.4174070358276367
Epoch  22  time  0.3938109874725342
Epoch  23  time  0.37001609802246094
Epoch  24  time  0.3698539733886719
Epoch  25  time  0.40327978134155273
Epoch  26  time  0.40351176261901855
Epoch  27  time  0.40015101432800293
Epoch  28  time  0.3831491470336914
Epoch  29  time  0.4310929775238037
Epoch  30  time  0.5770962238311768
Epoch  30 | Loss:     0.9770 | Accuracy:  98.00% | Correct:  49/50
Epoch  31  time  0.39865994453430176
Epoch  32  time  0.37341904640197754
Epoch  33  time  0.4190840721130371
Epoch  34  time  0.3907430171966553
Epoch  35  time  0.3869900703430176
Epoch  36  time  0.39249706268310547
Epoch  37  time  0.37486696243286133
Epoch  38  time  0.38773393630981445
Epoch  39  time  0.37876391410827637
Epoch  40  time  0.37739086151123047
Epoch  40 | Loss:     0.7802 | Accuracy:  98.00% | Correct:  49/50
Epoch  41  time  0.40010809898376465
Epoch  42  time  0.38652801513671875
Epoch  43  time  0.37795519828796387
Epoch  44  time  0.3714323043823242
Epoch  45  time  0.37934422492980957
Epoch  46  time  0.40614986419677734
Epoch  47  time  0.4050440788269043
Epoch  48  time  0.3736839294433594
Epoch  49  time  0.37200403213500977
Epoch  50  time  0.38141489028930664
Epoch  50 | Loss:     0.7243 | Accuracy:  98.00% | Correct:  49/50
Epoch  51  time  0.41335391998291016
Epoch  52  time  0.3991060256958008
Epoch  53  time  0.3736538887023926
Epoch  54  time  0.3704078197479248
Epoch  55  time  0.3973989486694336
Epoch  56  time  0.41052699089050293
Epoch  57  time  0.3924558162689209
Epoch  58  time  0.3743131160736084
Epoch  59  time  0.37747693061828613
Epoch  60  time  0.39292073249816895
Epoch  60 | Loss:     0.8949 | Accuracy:  98.00% | Correct:  49/50
Epoch  61  time  0.40917110443115234
Epoch  62  time  0.38897109031677246
Epoch  63  time  0.3787651062011719
Epoch  64  time  0.38068509101867676
Epoch  65  time  0.3774709701538086
Epoch  66  time  0.3854682445526123
Epoch  67  time  0.39637088775634766
Epoch  68  time  0.3804478645324707
Epoch  69  time  0.3779640197753906
Epoch  70  time  0.3780200481414795
Epoch  70 | Loss:     1.7293 | Accuracy:  98.00% | Correct:  49/50
Epoch  71  time  0.38405776023864746
Epoch  72  time  0.42041778564453125
Epoch  73  time  0.3918800354003906
Epoch  74  time  0.37038326263427734
Epoch  75  time  0.37511682510375977
Epoch  76  time  0.3926968574523926
Epoch  77  time  0.4158928394317627
Epoch  78  time  0.38059210777282715
Epoch  79  time  0.3693418502807617
Epoch  80  time  0.36841392517089844
Epoch  80 | Loss:     0.6865 | Accuracy:  98.00% | Correct:  49/50
Epoch  81  time  0.41436290740966797
Epoch  82  time  0.41202616691589355
Epoch  83  time  0.3767659664154053
Epoch  84  time  0.37226414680480957
Epoch  85  time  0.38390302658081055
Epoch  86  time  0.443972110748291
Epoch  87  time  0.3998589515686035
Epoch  88  time  0.37918829917907715
Epoch  89  time  0.37910985946655273
Epoch  90  time  0.3796696662902832
Epoch  90 | Loss:     0.5312 | Accuracy:  98.00% | Correct:  49/50
Epoch  91  time  0.3884117603302002
Epoch  92  time  0.3962068557739258
Epoch  93  time  0.38343286514282227
Epoch  94  time  0.38193202018737793
Epoch  95  time  0.372938871383667
Epoch  96  time  0.37612390518188477
Epoch  97  time  0.38796520233154297
Epoch  98  time  0.4058349132537842
Epoch  99  time  0.37926197052001953
Average time per epoch 0.4536869215965271 (for 100 epochs)

Results for Simple Dataset:
Total training time: 47.66 seconds
Average time per epoch: 0.4766 seconds
Final accuracy: 98.00%
Final loss: 0.5312

==================================================
Training on XOR Dataset
==================================================
Epoch  0  time  0.374495267868042
Epoch   0 | Loss:     6.7238 | Accuracy:  68.00% | Correct:  34/50
Epoch  1  time  0.37984418869018555
Epoch  2  time  0.4091191291809082
Epoch  3  time  0.40861988067626953
Epoch  4  time  0.3691439628601074
Epoch  5  time  0.38829588890075684
Epoch  6  time  0.4117701053619385
Epoch  7  time  0.429135799407959
Epoch  8  time  0.39523983001708984
Epoch  9  time  0.3732948303222656
Epoch  10  time  0.3737950325012207
Epoch  10 | Loss:     3.8573 | Accuracy:  70.00% | Correct:  35/50
Epoch  11  time  0.39743995666503906
Epoch  12  time  0.39690327644348145
Epoch  13  time  0.3893558979034424
Epoch  14  time  0.3699181079864502
Epoch  15  time  0.38465309143066406
Epoch  16  time  0.3860359191894531
Epoch  17  time  0.38306403160095215
Epoch  18  time  0.40139007568359375
Epoch  19  time  0.3867919445037842
Epoch  20  time  0.38155579566955566
Epoch  20 | Loss:     5.0975 | Accuracy:  94.00% | Correct:  47/50
Epoch  21  time  0.3784630298614502
Epoch  22  time  0.3786790370941162
Epoch  23  time  0.41443610191345215
Epoch  24  time  0.39619898796081543
Epoch  25  time  0.3737637996673584
Epoch  26  time  0.37627077102661133
Epoch  27  time  0.3944988250732422
Epoch  28  time  0.4189491271972656
Epoch  29  time  0.38364577293395996
Epoch  30  time  0.36948204040527344
Epoch  30 | Loss:     2.0469 | Accuracy:  94.00% | Correct:  47/50
Epoch  31  time  0.3706090450286865
Epoch  32  time  0.3898310661315918
Epoch  33  time  0.4187510013580322
Epoch  34  time  0.3830118179321289
Epoch  35  time  0.3721787929534912
Epoch  36  time  0.3716001510620117
Epoch  37  time  0.38704395294189453
Epoch  38  time  0.38007283210754395
Epoch  39  time  0.3673069477081299
Epoch  40  time  0.3750452995300293
Epoch  40 | Loss:     2.3789 | Accuracy:  96.00% | Correct:  48/50
Epoch  41  time  0.3807218074798584
Epoch  42  time  0.38724303245544434
Epoch  43  time  0.37608981132507324
Epoch  44  time  0.3656587600708008
Epoch  45  time  0.3768730163574219
Epoch  46  time  0.3888509273529053
Epoch  47  time  0.3790268898010254
Epoch  48  time  0.37302279472351074
Epoch  49  time  0.3798179626464844
Epoch  50  time  0.39730310440063477
Epoch  50 | Loss:     3.0984 | Accuracy:  96.00% | Correct:  48/50
Epoch  51  time  0.376209020614624
Epoch  52  time  0.3748009204864502
Epoch  53  time  0.374675989151001
Epoch  54  time  0.3850991725921631
Epoch  55  time  0.38521504402160645
Epoch  56  time  0.3768799304962158
Epoch  57  time  0.373276948928833
Epoch  58  time  0.39552903175354004
Epoch  59  time  0.42108798027038574
Epoch  60  time  0.4036848545074463
Epoch  60 | Loss:     1.7680 | Accuracy:  96.00% | Correct:  48/50
Epoch  61  time  0.4530770778656006
Epoch  62  time  0.5557749271392822
Epoch  63  time  0.4365060329437256
Epoch  64  time  0.4031260013580322
Epoch  65  time  0.40882372856140137
Epoch  66  time  0.3913569450378418
Epoch  67  time  0.5012631416320801
Epoch  68  time  0.5334022045135498
Epoch  69  time  1.1534380912780762
Epoch  70  time  0.9345331192016602
Epoch  70 | Loss:     1.2389 | Accuracy:  98.00% | Correct:  49/50
Epoch  71  time  0.5015342235565186
Epoch  72  time  0.3851592540740967
Epoch  73  time  0.39301609992980957
Epoch  74  time  0.41861510276794434
Epoch  75  time  0.4329700469970703
Epoch  76  time  0.37575817108154297
Epoch  77  time  0.3899509906768799
Epoch  78  time  0.41434597969055176
Epoch  79  time  0.5146708488464355
Epoch  80  time  0.5346908569335938
Epoch  80 | Loss:     1.6703 | Accuracy:  98.00% | Correct:  49/50
Epoch  81  time  0.469696044921875
Epoch  82  time  0.6198840141296387
Epoch  83  time  0.4211268424987793
Epoch  84  time  0.39420390129089355
Epoch  85  time  0.40061211585998535
Epoch  86  time  0.41120219230651855
Epoch  87  time  0.40193724632263184
Epoch  88  time  0.4093630313873291
Epoch  89  time  0.4018580913543701
Epoch  90  time  0.4795660972595215
Epoch  90 | Loss:     0.7249 | Accuracy:  98.00% | Correct:  49/50
Epoch  91  time  0.439162015914917
Epoch  92  time  0.40028905868530273
Epoch  93  time  0.40323925018310547
Epoch  94  time  0.39313507080078125
Epoch  95  time  0.38650989532470703
Epoch  96  time  0.3806490898132324
Epoch  97  time  0.397554874420166
Epoch  98  time  0.42981910705566406
Epoch  99  time  0.5611772537231445
Average time per epoch 0.41893765449523923 (for 100 epochs)

Results for XOR Dataset:
Total training time: 42.21 seconds
Average time per epoch: 0.4221 seconds
Final accuracy: 98.00%
Final loss: 0.7249

==================================================
Training on Split Dataset
==================================================
Epoch  0  time  0.5807287693023682
Epoch   0 | Loss:     7.4448 | Accuracy:  74.00% | Correct:  37/50
Epoch  1  time  0.45136189460754395
Epoch  2  time  0.43331384658813477
Epoch  3  time  0.39346909523010254
Epoch  4  time  0.3838200569152832
Epoch  5  time  0.6012551784515381
Epoch  6  time  0.6179289817810059
Epoch  7  time  0.42231202125549316
Epoch  8  time  0.39046692848205566
Epoch  9  time  0.4609031677246094
Epoch  10  time  0.4282259941101074
Epoch  10 | Loss:     5.2894 | Accuracy:  76.00% | Correct:  38/50
Epoch  11  time  0.4389159679412842
Epoch  12  time  0.3921980857849121
Epoch  13  time  0.39297008514404297
Epoch  14  time  0.3930199146270752
Epoch  15  time  0.4269077777862549
Epoch  16  time  0.3934597969055176
Epoch  17  time  0.3851010799407959
Epoch  18  time  0.42543911933898926
Epoch  19  time  0.41553401947021484
Epoch  20  time  0.4319431781768799
Epoch  20 | Loss:     4.5426 | Accuracy:  84.00% | Correct:  42/50
Epoch  21  time  0.38394999504089355
Epoch  22  time  0.38297104835510254
Epoch  23  time  0.4228782653808594
Epoch  24  time  0.6795408725738525
Epoch  25  time  0.47109508514404297
Epoch  26  time  0.4386861324310303
Epoch  27  time  0.40386104583740234
Epoch  28  time  0.39646220207214355
Epoch  29  time  0.41500329971313477
Epoch  30  time  0.423051118850708
Epoch  30 | Loss:     3.5540 | Accuracy:  80.00% | Correct:  40/50
Epoch  31  time  0.3965311050415039
Epoch  32  time  0.4013559818267822
Epoch  33  time  0.39864182472229004
Epoch  34  time  0.42054104804992676
Epoch  35  time  0.3880157470703125
Epoch  36  time  0.3991570472717285
Epoch  37  time  0.40883302688598633
Epoch  38  time  0.425199031829834
Epoch  39  time  0.8499531745910645
Epoch  40  time  0.4028160572052002
Epoch  40 | Loss:     4.3276 | Accuracy:  86.00% | Correct:  43/50
Epoch  41  time  0.407991886138916
Epoch  42  time  0.4191591739654541
Epoch  43  time  0.40951108932495117
Epoch  44  time  0.37209010124206543
Epoch  45  time  0.38110899925231934
Epoch  46  time  0.3939502239227295
Epoch  47  time  0.4254639148712158
Epoch  48  time  0.40836286544799805
Epoch  49  time  0.3799448013305664
Epoch  50  time  0.3780229091644287
Epoch  50 | Loss:     2.7223 | Accuracy:  86.00% | Correct:  43/50
Epoch  51  time  0.41956472396850586
Epoch  52  time  0.4752461910247803
Epoch  53  time  0.6390252113342285
Epoch  54  time  0.4266839027404785
Epoch  55  time  0.4331820011138916
Epoch  56  time  0.40189099311828613
Epoch  57  time  0.46663999557495117
Epoch  58  time  0.3951401710510254
Epoch  59  time  0.3905510902404785
Epoch  60  time  0.3926272392272949
Epoch  60 | Loss:     1.9164 | Accuracy:  96.00% | Correct:  48/50
Epoch  61  time  0.40936279296875
Epoch  62  time  0.40607714653015137
Epoch  63  time  0.40031909942626953
Epoch  64  time  0.3894839286804199
Epoch  65  time  0.3891890048980713
Epoch  66  time  0.789017915725708
Epoch  67  time  0.4437530040740967
Epoch  68  time  0.3841981887817383
Epoch  69  time  0.39371800422668457
Epoch  70  time  0.40567493438720703
Epoch  70 | Loss:     2.4044 | Accuracy:  98.00% | Correct:  49/50
Epoch  71  time  0.40444493293762207
Epoch  72  time  0.3831319808959961
Epoch  73  time  0.38307809829711914
Epoch  74  time  0.3838768005371094
Epoch  75  time  0.3889179229736328
Epoch  76  time  0.3810858726501465
Epoch  77  time  0.3779890537261963
Epoch  78  time  0.3735229969024658
Epoch  79  time  0.3981029987335205
Epoch  80  time  0.3944540023803711
Epoch  80 | Loss:     0.7058 | Accuracy:  92.00% | Correct:  46/50
Epoch  81  time  0.38875389099121094
Epoch  82  time  0.3821561336517334
Epoch  83  time  0.38314199447631836
Epoch  84  time  0.40670061111450195
Epoch  85  time  0.40074801445007324
Epoch  86  time  0.4219682216644287
Epoch  87  time  0.40555906295776367
Epoch  88  time  0.40154218673706055
Epoch  89  time  0.4079461097717285
Epoch  90  time  0.39425206184387207
Epoch  90 | Loss:     2.3930 | Accuracy:  98.00% | Correct:  49/50
Epoch  91  time  0.40503978729248047
Epoch  92  time  0.38845324516296387
Epoch  93  time  0.37816691398620605
Epoch  94  time  0.38249993324279785
Epoch  95  time  0.3994262218475342
Epoch  96  time  0.40210819244384766
Epoch  97  time  0.3860909938812256
Epoch  98  time  0.3958296775817871
Epoch  99  time  0.8333418369293213
Average time per epoch 0.42827099323272705 (for 100 epochs)

Results for Split Dataset:
Total training time: 43.11 seconds
Average time per epoch: 0.4311 seconds
Final accuracy: 98.00%
Final loss: 2.3930

==================================================
OVERALL TRAINING SUMMARY
==================================================

Simple Dataset:
- Final Accuracy: 98.00%
- Final Loss: 0.5312
- Average Epoch Time: 0.4766 seconds
- Total Training Time: 47.66 seconds

XOR Dataset:
- Final Accuracy: 98.00%
- Final Loss: 0.7249
- Average Epoch Time: 0.4221 seconds
- Total Training Time: 42.21 seconds

Split Dataset:
- Final Accuracy: 98.00%
- Final Loss: 2.3930
- Average Epoch Time: 0.4311 seconds
- Total Training Time: 43.11 seconds
```
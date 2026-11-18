**FAULT INJECTION RUN** (unsafe_reuse=2): a FAIL is the expected outcome.

# fault: gradient_equivalence (correctness)

**Status: FAIL** (execution FAIL; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- 0.6b-like.eager.float32.last2.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.last2.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.middle.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.middle.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.first_last.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.first_last.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.distributed.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.float32.distributed.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.last2.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.last2.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.middle.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.middle.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.first_last.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.first_last.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.distributed.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.eager.bfloat16.distributed.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.sdpa.float32.last2.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.sdpa.float32.last2.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.sdpa.float32.middle.b2.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- 0.6b-like.sdpa.float32.middle.b3.mb1: candidate refused: layer 2: the recomputed output differs from the forward's (model state changed between forward and backward?); refusing to return gradients
- ... 1388 more

- commit `e8440e41c84486e9a30cd1633ffc1029db5c190e` (tracked tree dirty: False), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU Quadro P1000
- config `/home/trulyaldi/qwen-grpo/research/l2l/configs/async_cpu.yaml` sha256 `f6c240cc9bd59f00`; fields used: ['model.dtype', 'execution']; backend `l2l_async`
- input sha256 `cb7569af732767ae`, tolerances sha256 `54e48e883478a48e`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like.eager.float32.final.b2 | loss | 2 | 0/2 | 0.024478555656969547 | 0.682130673109294 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | logits | 2 | 0/2 | 0.4908602833747864 | 1.5402199833601826 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | hidden_grads | 2 | 0/2 | 0.0009025516919791698 | 0.49548428823735613 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | grads | 28 | 0/28 | 0.15082520432770252 | 3.3385210621170174 | 0.0 |
| 0.6b-like.eager.float32.final.b3 | loss | 2 | 0/2 | 0.024478555656969547 | 0.682130673109294 | 0.0 |
| 0.6b-like.eager.float32.final.b3 | logits | 2 | 0/2 | 0.4908602833747864 | 1.5402199833601826 | 0.0 |
| 0.6b-like.eager.float32.final.b3 | hidden_grads | 2 | 0/2 | 0.0009025516919791698 | 0.49548428823735613 | 0.0 |
| 0.6b-like.eager.float32.final.b3 | grads | 28 | 0/28 | 0.15082520432770252 | 3.3385210621170174 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | loss | 2 | 0/2 | 0.023961852304637432 | 0.6626122911788921 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | logits | 2 | 0/2 | 0.4873046875 | 1.521341463414634 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | hidden_grads | 2 | 0/2 | 0.0009307861328125 | 0.5104602510460251 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | grads | 28 | 0/28 | 0.16048831399530172 | 3.572976244602271 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b3 | loss | 2 | 0/2 | 0.023961852304637432 | 0.6626122911788921 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b3 | logits | 2 | 0/2 | 0.4873046875 | 1.521341463414634 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b3 | hidden_grads | 2 | 0/2 | 0.0009307861328125 | 0.5104602510460251 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b3 | grads | 28 | 0/28 | 0.16048831399530172 | 3.572976244602271 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | loss | 2 | 0/2 | 0.02447857614606619 | 0.6821301110699445 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | logits | 2 | 0/2 | 0.490860253572464 | 1.5402197458149844 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | hidden_grads | 2 | 0/2 | 0.0009025499457493424 | 0.49548348792071084 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | grads | 28 | 0/28 | 0.15082534961402416 | 3.33852235098621 | 0.0 |
| 0.6b-like.sdpa.float32.final.b3 | loss | 2 | 0/2 | 0.02447857614606619 | 0.6821301110699445 | 0.0 |
| 0.6b-like.sdpa.float32.final.b3 | logits | 2 | 0/2 | 0.490860253572464 | 1.5402197458149844 | 0.0 |
| 0.6b-like.sdpa.float32.final.b3 | hidden_grads | 2 | 0/2 | 0.0009025499457493424 | 0.49548348792071084 | 0.0 |
| 0.6b-like.sdpa.float32.final.b3 | grads | 28 | 0/28 | 0.15082534961402416 | 3.33852235098621 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | loss | 2 | 0/2 | 0.024469981901347637 | 0.6758925881398054 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | logits | 2 | 0/2 | 0.4853515625 | 1.5152439024390243 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | hidden_grads | 2 | 0/2 | 0.0009613037109375 | 0.5316455696202531 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | grads | 28 | 0/28 | 0.16263694316148758 | 3.5991560711835753 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b3 | loss | 2 | 0/2 | 0.024469981901347637 | 0.6758925881398054 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b3 | logits | 2 | 0/2 | 0.4853515625 | 1.5152439024390243 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b3 | hidden_grads | 2 | 0/2 | 0.0009613037109375 | 0.5316455696202531 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b3 | grads | 28 | 0/28 | 0.16263694316148758 | 3.5991560711835753 | 0.0 |
| 4b-like.eager.float32.final.b2 | loss | 2 | 0/2 | 0.019306926056742668 | 0.5474472019279674 | 0.0 |
| 4b-like.eager.float32.final.b2 | logits | 2 | 0/2 | 0.5083459317684174 | 1.31664209723382 | 0.0 |
| 4b-like.eager.float32.final.b2 | hidden_grads | 2 | 0/2 | 0.0014702382031828165 | 0.9330533268318632 | 0.0 |
| 4b-like.eager.float32.final.b2 | grads | 28 | 0/28 | 0.1073282491415739 | 2.2926921258264747 | 0.0 |
| 4b-like.eager.float32.final.b3 | loss | 2 | 0/2 | 0.019306926056742668 | 0.5474472019279674 | 0.0 |
| 4b-like.eager.float32.final.b3 | logits | 2 | 0/2 | 0.5083459317684174 | 1.31664209723382 | 0.0 |
| 4b-like.eager.float32.final.b3 | hidden_grads | 2 | 0/2 | 0.0014702382031828165 | 0.9330533268318632 | 0.0 |
| 4b-like.eager.float32.final.b3 | grads | 28 | 0/28 | 0.1073282491415739 | 2.2926921258264747 | 0.0 |
| 4b-like.eager.float32.last2.b2 | loss | 2 | 0/2 | 0.008861303329467773 | 0.06238709052019876 | 0.0 |
| 4b-like.eager.float32.last2.b2 | logits | 2 | 0/2 | 0.5600068867206573 | 1.242052504108235 | 0.0 |
| 4b-like.eager.float32.last2.b2 | hidden_grads | 4 | 0/4 | 0.0006066264731998672 | 0.475913162696265 | 0.0 |
| 4b-like.eager.float32.last2.b2 | grads | 56 | 0/56 | 0.058902574819512665 | 0.8934163452291353 | 0.0 |
| 4b-like.eager.float32.last2.b3 | loss | 2 | 0/2 | 0.008861303329467773 | 0.06238709052019876 | 0.0 |
| 4b-like.eager.float32.last2.b3 | logits | 2 | 0/2 | 0.5600068867206573 | 1.242052504108235 | 0.0 |
| 4b-like.eager.float32.last2.b3 | hidden_grads | 4 | 0/4 | 0.0006066264731998672 | 0.475913162696265 | 0.0 |
| 4b-like.eager.float32.last2.b3 | grads | 56 | 0/56 | 0.058902574819512665 | 0.8934163452291353 | 0.0 |
| 4b-like.eager.float32.middle.b2 | loss | 2 | 0/2 | 0.010341694578528404 | 0.35304355489674016 | 0.0 |
| 4b-like.eager.float32.middle.b2 | logits | 2 | 0/2 | 0.7548729777336121 | 1.8610948933534461 | 0.0 |
| 4b-like.eager.float32.middle.b2 | hidden_grads | 6 | 0/6 | 0.0020089438767172396 | 2.1543932630237466 | 0.0 |
| 4b-like.eager.float32.middle.b2 | grads | 28 | 0/28 | 0.15028649382293224 | 6.152829054882071 | 0.0 |
| 4b-like.eager.float32.middle.b3 | loss | 2 | 0/2 | 0.010341694578528404 | 0.35304355489674016 | 0.0 |
| 4b-like.eager.float32.middle.b3 | logits | 2 | 0/2 | 0.7548729777336121 | 1.8610948933534461 | 0.0 |
| 4b-like.eager.float32.middle.b3 | hidden_grads | 6 | 0/6 | 0.0020089438767172396 | 2.1543932630237466 | 0.0 |
| 4b-like.eager.float32.middle.b3 | grads | 28 | 0/28 | 0.15028649382293224 | 6.152829054882071 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | loss | 2 | 0/2 | 0.019553542137145996 | 0.557059481782229 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | logits | 2 | 0/2 | 0.513671875 | 1.3350253807106598 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | hidden_grads | 2 | 0/2 | 0.00146484375 | 0.9230769230769231 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | grads | 28 | 0/28 | 0.10481388494372368 | 2.3397241370822455 | 0.0 |
| 4b-like.eager.bfloat16.final.b3 | loss | 2 | 0/2 | 0.019553542137145996 | 0.557059481782229 | 0.0 |
| 4b-like.eager.bfloat16.final.b3 | logits | 2 | 0/2 | 0.513671875 | 1.3350253807106598 | 0.0 |
| 4b-like.eager.bfloat16.final.b3 | hidden_grads | 2 | 0/2 | 0.00146484375 | 0.9230769230769231 | 0.0 |
| 4b-like.eager.bfloat16.final.b3 | grads | 28 | 0/28 | 0.10481388494372368 | 2.3397241370822455 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | loss | 2 | 0/2 | 0.008806183934211731 | 0.06203742738874953 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | logits | 2 | 0/2 | 0.5634765625 | 1.2435344827586208 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | hidden_grads | 4 | 0/4 | 0.0006072521209716797 | 0.4766092814371258 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | grads | 56 | 0/56 | 0.05937616934534162 | 0.8986176085208015 | 0.0 |
| 4b-like.eager.bfloat16.last2.b3 | loss | 2 | 0/2 | 0.008806183934211731 | 0.06203742738874953 | 0.0 |
| 4b-like.eager.bfloat16.last2.b3 | logits | 2 | 0/2 | 0.5634765625 | 1.2435344827586208 | 0.0 |
| 4b-like.eager.bfloat16.last2.b3 | hidden_grads | 4 | 0/4 | 0.0006072521209716797 | 0.4766092814371258 | 0.0 |
| 4b-like.eager.bfloat16.last2.b3 | grads | 56 | 0/56 | 0.05937616934534162 | 0.8986176085208015 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | loss | 2 | 0/2 | 0.010098734870553017 | 0.3438583889082821 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | logits | 2 | 0/2 | 0.75390625 | 1.8557692307692308 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | hidden_grads | 6 | 0/6 | 0.002109527587890625 | 2.2757201646090537 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | grads | 28 | 0/28 | 0.15429200697690248 | 6.275997141229707 | 0.0 |
| 4b-like.eager.bfloat16.middle.b3 | loss | 2 | 0/2 | 0.010098734870553017 | 0.3438583889082821 | 0.0 |
| 4b-like.eager.bfloat16.middle.b3 | logits | 2 | 0/2 | 0.75390625 | 1.8557692307692308 | 0.0 |
| 4b-like.eager.bfloat16.middle.b3 | hidden_grads | 6 | 0/6 | 0.002109527587890625 | 2.2757201646090537 | 0.0 |
| 4b-like.eager.bfloat16.middle.b3 | grads | 28 | 0/28 | 0.15429200697690248 | 6.275997141229707 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | loss | 2 | 0/2 | 0.019306888803839684 | 0.5474464347583596 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | logits | 2 | 0/2 | 0.5083459168672562 | 1.3166420586390464 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | hidden_grads | 2 | 0/2 | 0.0014702384360134602 | 0.9330534056582439 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | grads | 28 | 0/28 | 0.10732830129563808 | 2.292693057467964 | 0.0 |
| 4b-like.sdpa.float32.final.b3 | loss | 2 | 0/2 | 0.019306888803839684 | 0.5474464347583596 | 0.0 |
| 4b-like.sdpa.float32.final.b3 | logits | 2 | 0/2 | 0.5083459168672562 | 1.3166420586390464 | 0.0 |
| 4b-like.sdpa.float32.final.b3 | hidden_grads | 2 | 0/2 | 0.0014702384360134602 | 0.9330534056582439 | 0.0 |
| 4b-like.sdpa.float32.final.b3 | grads | 28 | 0/28 | 0.10732830129563808 | 2.292693057467964 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | loss | 2 | 0/2 | 0.008861303329467773 | 0.06238709052019876 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | logits | 2 | 0/2 | 0.4990319162607193 | 1.2514782365242714 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | hidden_grads | 4 | 0/4 | 0.0006066266541893128 | 0.47591330468686927 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | grads | 56 | 0/56 | 0.058902641758322716 | 0.8934171586098284 | 0.0 |
| 4b-like.sdpa.float32.last2.b3 | loss | 2 | 0/2 | 0.008861303329467773 | 0.06238709052019876 | 0.0 |
| 4b-like.sdpa.float32.last2.b3 | logits | 2 | 0/2 | 0.4990319162607193 | 1.2514782365242714 | 0.0 |
| 4b-like.sdpa.float32.last2.b3 | hidden_grads | 4 | 0/4 | 0.0006066266541893128 | 0.47591330468686927 | 0.0 |
| 4b-like.sdpa.float32.last2.b3 | grads | 56 | 0/56 | 0.058902641758322716 | 0.8934171586098284 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | loss | 2 | 0/2 | 0.010341715067625046 | 0.35304400741293945 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | logits | 2 | 0/2 | 0.7548728287220001 | 1.8610942524830432 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | hidden_grads | 6 | 0/6 | 0.0020089466124773026 | 2.154396196855357 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | grads | 28 | 0/28 | 0.1502864481881261 | 6.152824371352775 | 0.0 |
| 4b-like.sdpa.float32.middle.b3 | loss | 2 | 0/2 | 0.010341715067625046 | 0.35304400741293945 | 0.0 |
| 4b-like.sdpa.float32.middle.b3 | logits | 2 | 0/2 | 0.7548728287220001 | 1.8610942524830432 | 0.0 |
| 4b-like.sdpa.float32.middle.b3 | hidden_grads | 6 | 0/6 | 0.0020089466124773026 | 2.154396196855357 | 0.0 |
| 4b-like.sdpa.float32.middle.b3 | grads | 28 | 0/28 | 0.1502864481881261 | 6.152824371352775 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | loss | 2 | 0/2 | 0.019762545824050903 | 0.5577901409445878 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | logits | 2 | 0/2 | 0.5146484375 | 1.3375634517766497 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | hidden_grads | 2 | 0/2 | 0.00147247314453125 | 0.9323671497584541 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | grads | 28 | 0/28 | 0.10671944729983807 | 2.36603471719157 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b3 | loss | 2 | 0/2 | 0.019762545824050903 | 0.5577901409445878 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b3 | logits | 2 | 0/2 | 0.5146484375 | 1.3375634517766497 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b3 | hidden_grads | 2 | 0/2 | 0.00147247314453125 | 0.9323671497584541 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b3 | grads | 28 | 0/28 | 0.10671944729983807 | 2.36603471719157 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | loss | 2 | 0/2 | 0.00910872220993042 | 0.06423112596761586 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | logits | 2 | 0/2 | 0.4970703125 | 1.2475490196078431 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | hidden_grads | 4 | 0/4 | 0.0006030797958374023 | 0.4761859939759036 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | grads | 56 | 0/56 | 0.059545101248659194 | 0.9063343432575632 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b3 | loss | 2 | 0/2 | 0.00910872220993042 | 0.06423112596761586 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b3 | logits | 2 | 0/2 | 0.4970703125 | 1.2475490196078431 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b3 | hidden_grads | 4 | 0/4 | 0.0006030797958374023 | 0.4761859939759036 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b3 | grads | 56 | 0/56 | 0.059545101248659194 | 0.9063343432575632 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | loss | 2 | 0/2 | 0.009887894615530968 | 0.3387867650812906 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | logits | 2 | 0/2 | 0.751953125 | 1.8509615384615385 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | hidden_grads | 6 | 0/6 | 0.002048492431640625 | 2.2098765432098766 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | grads | 28 | 0/28 | 0.15140022058039904 | 6.276452877632512 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b3 | loss | 2 | 0/2 | 0.009887894615530968 | 0.3387867650812906 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b3 | logits | 2 | 0/2 | 0.751953125 | 1.8509615384615385 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b3 | hidden_grads | 6 | 0/6 | 0.002048492431640625 | 2.2098765432098766 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b3 | grads | 28 | 0/28 | 0.15140022058039904 | 6.276452877632512 | 0.0 |


measurements: `{"saved_bytes": {"0.6b-like.eager.float32.final.b2": 92080, "0.6b-like.eager.float32.final.b3": 92080, "0.6b-like.eager.float32.last2.b2": 69336, "0.6b-like.eager.float32.last2.b3": 69336, "0.6b-like.eager.float32.middle.b2": 49880, "0.6b-like.eager.float32.middle.b3": 49880, "0.6b-like.eager.float32.first_last.b2": 77016, "0.6b-like.eager.float32.first_last.b3": 77016, "0.6b-like.eager.float32.distributed.b2": 115928, "0.6b-like.eager.float32.distributed.b3": 115928, "0.6b-like.eager.bfloat16.final.b2": 79920, "0.6b-like.eager.bfloat16.final.b3": 79920, "0.6b-like.eager.bfloat16.last2.b2": 61336, "0.6b-like.eager.bfloat16.last2.b3": 61336, "0.6b-like.eager.bfloat16.middle.b2": 41880, "0.6b-like.eager.bfloat16.middle.b3": 41880, "0.6b-like.eager.bfloat16.first_last.b2": 65176, "0.6b-like.eager.bfloat16.first_last.b3": 65176, "0.6b-like.eager.bfloat16.distributed.b2": 104088, "0.6b-like.eager.bfloat16.distributed.b3": 104088, "0.6b-like.sdpa.float32.final.b2": 92080, "0.6b-like.sdpa.float32.final.b3": 92080, "0.6b-like.sdpa.float32.last2.b2": 69336, "0.6b-like.sdpa.float32.last2.b3": 69336, "0.6b-like.sdpa.float32.middle.b2": 49880, "0.6b-like.sdpa.float32.middle.b3": 49880, "0.6b-like.sdpa.float32.first_last.b2": 77016, "0.6b-like.sdpa.float32.first_last.b3": 77016, "0.6b-like.sdpa.float32.distributed.b2": 115928, "0.6b-like.sdpa.float32.distributed.b3": 115928, "0.6b-like.sdpa.bfloat16.final.b2": 79920, "0.6b-like.sdpa.bfloat16.final.b3": 79920, "0.6b-like.sdpa.bfloat16.last2.b2": 61336, "0.6b-like.sdpa.bfloat16.last2.b3": 61336, "0.6b-like.sdpa.bfloat16.middle.b2": 41880, "0.6b-like.sdpa.bfloat16.middle.b3": 41880, "0.6b-like.sdpa.bfloat16.first_last.b2": 65176, "0.6b-like.sdpa.bfloat16.first_last.b3": 65176, "0.6b-like.sdpa.bfloat16.distributed.b2": 104088, "0.6b-like.sdpa.bfloat16.distributed.b3": 104088, "4b-like.eager.float32.final.b2": 109680, "4b-like.eager.float32.final.b3": 109680, "4b-like.eager.float32.last2.b2": 165872, "4b-like.eager.float32.last2.b3": 165872, "4b-like.eager.float32.middle.b2": 128880, "4b-like.eager.float32.middle.b3": 128880, "4b-like.eager.float32.first_last.b2": 102136, "4b-like.eager.float32.first_last.b3": 102136, "4b-like.eager.float32.distributed.b2": 148728, "4b-like.eager.float32.distributed.b3": 148728, "4b-like.eager.bfloat16.final.b2": 94480, "4b-like.eager.bfloat16.final.b3": 94480, "4b-like.eager.bfloat16.last2.b2": 145872, "4b-like.eager.bfloat16.last2.b3": 145872, "4b-like.eager.bfloat16.middle.b2": 104080, "4b-like.eager.bfloat16.middle.b3": 104080, "4b-like.eager.bfloat16.first_last.b2": 82536, "4b-like.eager.bfloat16.first_last.b3": 82536, "4b-like.eager.bfloat16.distributed.b2": 129128, "4b-like.eager.bfloat16.distributed.b3": 129128, "4b-like.sdpa.float32.final.b2": 109680, "4b-like.sdpa.float32.final.b3": 109680, "4b-like.sdpa.float32.last2.b2": 165872, "4b-like.sdpa.float32.last2.b3": 165872, "4b-like.sdpa.float32.middle.b2": 128880, "4b-like.sdpa.float32.middle.b3": 128880, "4b-like.sdpa.float32.first_last.b2": 102136, "4b-like.sdpa.float32.first_last.b3": 102136, "4b-like.sdpa.float32.distributed.b2": 148728, "4b-like.sdpa.float32.distributed.b3": 148728, "4b-like.sdpa.bfloat16.final.b2": 94480, "4b-like.sdpa.bfloat16.final.b3": 94480, "4b-like.sdpa.bfloat16.last2.b2": 145872, "4b-like.sdpa.bfloat16.last2.b3": 145872, "4b-like.sdpa.bfloat16.middle.b2": 104080, "4b-like.sdpa.bfloat16.middle.b3": 104080, "4b-like.sdpa.bfloat16.first_last.b2": 82536, "4b-like.sdpa.bfloat16.first_last.b3": 82536, "4b-like.sdpa.bfloat16.distributed.b2": 129128, "4b-like.sdpa.bfloat16.distributed.b3": 129128}, "resident_saved_bytes": {"0.6b-like.eager.float32.final": 418384, "0.6b-like.eager.float32.last2": 846288, "0.6b-like.eager.float32.middle": 724816, "0.6b-like.eager.float32.first_last": 1459152, "0.6b-like.eager.float32.distributed": 1702096, "0.6b-like.eager.bfloat16.final": 318480, "0.6b-like.eager.bfloat16.last2": 638288, "0.6b-like.eager.bfloat16.middle": 516816, "0.6b-like.eager.bf ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 730.8125 MiB, high-water mark 730.8125 MiB (this process)

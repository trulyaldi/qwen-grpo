**FAULT INJECTION RUN** (stale_position): a FAIL is the expected outcome.

# t06_generation_equivalence_stale_position: generation_equivalence (correctness)

**Status: FAIL** (execution PASS; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.1: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.2: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.3: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.4: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.5: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.6: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.7: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.kv_cpu.resident: max_abs 4.300e+01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.1: max_abs 8.364e-02 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.2: max_abs 1.774e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.3: max_abs 3.357e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.4: max_abs 4.204e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.5: max_abs 3.274e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.6: max_abs 2.992e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.7: max_abs 3.806e-01 > bound 0.000e+00
- greedy_sequences.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed: max_abs 4.300e+01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.1: max_abs 8.364e-02 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.2: max_abs 1.774e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.3: max_abs 3.357e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.4: max_abs 4.204e-01 > bound 0.000e+00
- ... 857 more

- commit `04e56b95bb3e64721c4d894a664cd2caa0c160c6` (tracked tree dirty: False), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/store_cpu.yaml` sha256 `cd9f52c56460a60b`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `5948041fac234ce6`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| None | greedy_sequences | 104 | 0/104 | n/a | None | None |
| 0.6b-like.eager.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 43.0 | 0.6935483870967742 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | cached_logits | 16 | 2/16 | 0.42044998705387115 | 1.1862043825447535 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | prefix_logits | 8 | 1/8 | 0.5422794222831726 | 1.4074793844580158 | 3.852840960025788e-06 |
| 0.6b-like.eager.float32.V64.p9 | prefix_probs | 8 | 1/8 | 0.009256714023649693 | 0.41571347872811093 | 8.906821161508561e-06 |
| 0.6b-like.eager.float32.V64.p9 | sampled_probs | 16 | 2/16 | 0.009168440476059914 | 0.40274076211701115 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | cached_logits | 16 | 2/16 | 0.37970712780952454 | 1.1402372410978654 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | prefix_logits | 8 | 1/8 | 0.316610723733902 | 1.0270072825466263 | 3.082847893238068e-06 |
| 0.6b-like.eager.float32.V64.p17 | prefix_probs | 8 | 1/8 | 0.005232164636254311 | 0.24614142886259754 | 8.502696454524994e-06 |
| 0.6b-like.eager.float32.V64.p17 | sampled_probs | 16 | 2/16 | 0.00701740849763155 | 0.304557363274696 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 43.0 | 0.6935483870967742 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | cached_logits | 16 | 2/16 | 0.4169921875 | 1.1730769230769231 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 1/8 | 0.5322265625 | 1.3762626262626263 | 0.00604248046875 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.009205690585076809 | 0.4146224698450208 | 0.014431680180132389 |
| 0.6b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 2/16 | 0.009062206372618675 | 0.3985826218178422 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | cached_logits | 16 | 2/16 | 0.38671875 | 1.106145251396648 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 1/8 | 0.3154296875 | 0.95 | 0.00518798828125 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.005209370516240597 | 0.24083269488954198 | 0.014059930015355349 |
| 0.6b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 2/16 | 0.0070173246785998344 | 0.30452191991037475 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 43.0 | 0.6935483870967742 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | cached_logits | 16 | 2/16 | 0.4204498678445816 | 1.1862042456960447 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 1/8 | 0.5422793477773666 | 1.4074789733377753 | 3.852841556072235e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 1/8 | 0.009256714954972267 | 0.41571352055324956 | 8.906821161508561e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 2/16 | 0.009168428368866444 | 0.40274026323833684 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | cached_logits | 16 | 2/16 | 0.3797071576118469 | 1.1402373305924136 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 1/8 | 0.31661076843738556 | 1.0270077254005532 | 3.0828469991683964e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 1/8 | 0.005232164636254311 | 0.24614142886259754 | 8.502696454524994e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 2/16 | 0.007017410360276699 | 0.30455744411398245 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 43.0 | 0.6935483870967742 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | cached_logits | 16 | 2/16 | 0.41357421875 | 1.1829608938547487 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 1/8 | 0.5283203125 | 1.511173184357542 | 0.005462646484375 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.009250815957784653 | 0.41575165336089553 | 0.014463034179061652 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 2/16 | 0.009196812286973 | 0.404510874970568 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | cached_logits | 16 | 2/16 | 0.3798828125 | 1.1441176470588235 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 1/8 | 0.3134765625 | 1.0158227848101267 | 0.00482177734375 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.005176662467420101 | 0.24343975211045538 | 0.013822026085108519 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 2/16 | 0.007072163745760918 | 0.3069118972728707 | 0.0 |
| 4b-like.eager.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 41.0 | 0.6612903225806451 | 0.0 |
| 4b-like.eager.float32.V64.p9 | cached_logits | 16 | 2/16 | 0.45794132351875305 | 1.2393628145850935 | 0.0 |
| 4b-like.eager.float32.V64.p9 | prefix_logits | 8 | 1/8 | 0.6758248955011368 | 1.5394806167801327 | 4.389953911304474e-06 |
| 4b-like.eager.float32.V64.p9 | prefix_probs | 8 | 1/8 | 0.011798455379903316 | 0.48900620513356163 | 9.65096578001976e-06 |
| 4b-like.eager.float32.V64.p9 | sampled_probs | 16 | 2/16 | 0.010875190608203411 | 0.4711341662825662 | 0.0 |
| 4b-like.eager.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 51.0 | 0.8225806451612904 | 0.0 |
| 4b-like.eager.float32.V64.p17 | cached_logits | 16 | 2/16 | 0.6062021553516388 | 1.2152945549931662 | 0.0 |
| 4b-like.eager.float32.V64.p17 | prefix_logits | 8 | 1/8 | 0.6762652695178986 | 1.6552985799636168 | 4.085457921028138e-06 |
| 4b-like.eager.float32.V64.p17 | prefix_probs | 8 | 1/8 | 0.009753800928592682 | 0.4284724218947783 | 9.105651080608369e-06 |
| 4b-like.eager.float32.V64.p17 | sampled_probs | 16 | 2/16 | 0.011245832778513432 | 0.4838544497053266 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 36.0 | 0.5806451612903226 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | cached_logits | 16 | 2/16 | 0.69921875 | 2.1183431952662723 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 1/8 | 0.3623046875 | 0.653169014084507 | 0.0086669921875 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0058389827609062195 | 0.2171137237692978 | 0.01748087927699089 |
| 4b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 2/16 | 0.010649507865309715 | 0.4604577234438602 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 51.0 | 0.8225806451612904 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | cached_logits | 16 | 2/16 | 0.61328125 | 1.236220472440945 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 1/8 | 0.669921875 | 1.6570048309178744 | 0.006317138671875 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.009666969999670982 | 0.4240780420968494 | 0.014816920179873705 |
| 4b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 2/16 | 0.011353584937751293 | 0.4864408243193538 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 41.0 | 0.6612903225806451 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | cached_logits | 16 | 2/16 | 0.4579412043094635 | 1.239362491959567 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 1/8 | 0.6758243888616562 | 1.53947956720325 | 4.38995361328125e-06 |
| 4b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 1/8 | 0.01179844792932272 | 0.4890059340835459 | 9.6509650349617e-06 |
| 4b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 2/16 | 0.010875191539525986 | 0.47113428266407587 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 51.0 | 0.8225806451612904 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | cached_logits | 16 | 2/16 | 0.6062020435929298 | 1.2152947666022795 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 1/8 | 0.6762656569480896 | 1.6552979585345917 | 4.085461795330048e-06 |
| 4b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 1/8 | 0.00975380465388298 | 0.42847269071984007 | 9.105648845434189e-06 |
| 4b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 2/16 | 0.011245837435126305 | 0.48385457250422 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 36.0 | 0.5806451612903226 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | cached_logits | 16 | 2/16 | 0.828125 | 1.7377049180327868 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 1/8 | 0.36328125 | 0.6595744680851063 | 0.00860595703125 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0058672018349170685 | 0.21905875801077432 | 0.017409398406744003 |
| 4b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 2/16 | 0.010543217882514 | 0.45506767591166536 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 51.0 | 0.8225806451612904 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | cached_logits | 16 | 2/16 | 0.6064453125 | 1.2176470588235293 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 1/8 | 0.673828125 | 1.6428571428571428 | 0.00640869140625 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.009694167412817478 | 0.42623395546886855 | 0.014783451054245235 |
| 4b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 2/16 | 0.011331088840961456 | 0.4864261098453812 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | greedy_sequences | 2 | 0/2 | 48.0 | 0.7741935483870968 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | cached_logits | 16 | 2/16 | 4.089366436004639 | 2.6688995611389847 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_logits | 8 | 1/8 | 2.7606146335601807 | 0.6298324969731874 | 4.383093357086182e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_probs | 8 | 1/8 | 0.1715231668204069 | 8.001301867466886 | 8.5747629404068e-06 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | sampled_probs | 16 | 2/16 | 0.1380183957517147 | 3.6974752419863735 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | greedy_sequences | 2 | 0/2 | 32.0 | 0.5161290322580645 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | cached_logits | 16 | 2/16 | 4.738409042358398 | 2.037283706191922 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_logits | 8 | 1/8 | 4.7384092807769775 | 2.0372835998617367 | 2.3258466720581057e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_probs | 8 | 1/8 | 0.14499897882342339 | 0.8503046103300818 | 6.8210369348526e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | sampled_probs | 16 | 2/16 | 0.2550369221717119 | 0.8990555674797609 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | greedy_sequences | 2 | 0/2 | 96773.0 | 0.6380581269615212 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | cached_logits | 16 | 2/16 | 0.662109375 | 1.107843137254902 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_logits | 8 | 1/8 | 0.4736328125 | 0.8981481481481481 | 0.00823974609375 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_probs | 8 | 8/8 | 3.2305633794749156e-06 | 0.29960896936998427 | 7.00868936291954e-06 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | sampled_probs | 16 | 2/16 | 5.11413736603572e-06 | 0.4596311133707814 | 0.0 |


measurements: `{"weight_loads": {"0.6b-like.eager.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "4b-like.eager.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "0.6b-like.sdpa.float32.V64.eos.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.eos.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V151936.chat.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}}, "near_ties": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 6, "0.6b-like.eager.bfloat16.V64.p17": 5, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 4, "0.6b-like.sdpa.bfloat16.V64.p17": 5, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 0, "4b-like.eager.bfloat16.V64.p9": 7, "4b-like.eager.bfloat16.V64.p17": 8, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 0, "4b-like.sdpa.bfloat16.V64.p9": 7, "4b-like.sdpa.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.eos.p9": 1, "0.6b-like.sdpa.float32.V64.eos.p17": 1, "0.6b-like.sdpa.bfloat16.V151936.chat": 12}, "greedy_steps": {"0.6b-like.eager.float32.V64.p9": 8, "0.6b-like.eager.float32.V64.p17": 8, "0.6b-like.eager.bfloat16.V64.p9": 8, "0.6b-like.eager.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.p9": 8, "0.6b-like.sdpa.float32.V64.p17": 8, "0.6b-like.sdpa.bfloat16.V64.p9": 8, "0.6b-like.sdpa.bfloat16.V64.p17": 8, "4b-like.eager.float32.V64.p9": 8, "4b-like.eager.float32.V64.p17": 8, "4b-like.eager.bfloat16.V64.p9": 8, "4b-like.eager.bfloat16.V64.p17": 8, "4b-like.sdpa.float32.V64.p9": 8, "4b-like.sdpa.float32.V64.p17": 8, "4b-like.sdpa.bfloat16.V64.p9": 8, "4b-like.sdpa.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.eos.p9": 8, "0.6b-like.sdpa.float32.V64.eos.p17": 8, "0.6b-like.sdpa.bfloat16.V151936.chat": 8}, "eos_padded_tail_rows": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 0, "0.6b-like.eager.bfloat16.V64.p17": 0, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 0, "0.6b-like.sdpa.bfloat16.V64.p17": 0, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 0, "4b-like.eager.bfloat16.V64.p9": 0, "4b-like.eager.bfloat16.V64.p17": 0, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 0, "4b-like.sdpa.bfloat16.V64.p9": 0, "4b-like.sdpa.bfloat16.V64.p17": 0, "0.6b-like.sdpa.float32.V64.eos.p9": 2, "0.6b-like.sdpa.float32.V64.eos.p17": 2, "0.6b-like.sdpa.bfloat16.V151936.chat": 0}, "same_seed_samples_identical": {"0.6b-like.eager.float32.V64.p9": false, "0.6b-like.eager.float32.V64.p17": false, "0.6b-like.eager.bfloat16.V64.p9": false, "0.6b-like.eager.bfloat16.V64.p17": fal ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 904.44140625 MiB, high-water mark 996.96484375 MiB (this process)

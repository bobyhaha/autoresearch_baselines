# Evidence memory

## b0000: baseline

Locked baseline established at val_bpb=0.991068.

## t0001: keep

t0001 improved the prior champion: val_bpb=0.983511, delta=-0.007557. Hypothesis: At 50M params and ~1019 optimizer steps, the 2**19-token (524K) batch sits above the critical batch size for this loss regime, so gradient tokens are spent widening each batch instead of taking more Muon/AdamW steps; halving TOTAL_BATCH_SIZE to 2**18 should roughly double the step count at the same token throughput and reach a lower val_bpb within the same 300 seconds.

## t0002: keep

t0002 improved the prior champion: val_bpb=0.981629, delta=-0.001882. Hypothesis: Attention accounts for about 26 percent of the estimated FLOPs per token in this 8-layer SSSL configuration (six short layers at a 1024 window plus two full-context layers at 2048). Because every fourth layer is already full-context, the short layers are doing local work that a 512-token span can still cover, so halving the short window from sequence_len//2 to sequence_len//4 should cut roughly 8 percent of per-token FLOPs and convert that directly into additional optimizer steps inside the fixed 300 seconds, lowering val_bpb.

## t0003: discard

t0003 did not improve the prior champion: val_bpb=0.986688, delta=+0.005059. Hypothesis: The batch-size lever is not yet saturated. Halving TOTAL_BATCH_SIZE from 2**19 to 2**18 in t0001 bought 0.007557 bpb by doubling optimizer steps, which means the previous batch was above the critical batch size for this loss regime. If the critical size is still below 2**18, a further halving to 2**17 (131K tokens per step, with DEVICE_BATCH_SIZE reduced to 64 so gradient accumulation stays at 1) should again convert batch width into steps and lower val_bpb; if the critical size has been crossed, the added gradient noise will make it rise, which locates the optimum.

## t0004: keep

t0004 improved the prior champion: val_bpb=0.981147, delta=-0.000482. Hypothesis: The learning rates in this recipe were tuned at TOTAL_BATCH_SIZE=2**19, but the champion now trains at 2**18 with those rates unchanged, so the update-size to gradient-noise ratio is roughly sqrt(2) too aggressive. This also offers a competing explanation for the t0003 failure: at 2**17 the inherited rates were about 2x too large, so t0003 may have measured a learning-rate mismatch rather than a true critical batch size. Rescaling the three main learning rates by 1/sqrt(2), the standard square-root batch scaling, should lower val_bpb below champion t0002.

## t0005: discard

t0005 did not improve the prior champion: val_bpb=0.981248, delta=+0.000101. Hypothesis: t0003 and t0004 offer competing explanations for the failure at TOTAL_BATCH_SIZE=2**17. Either 2**18 really is the critical batch size, or t0003 simply inherited learning rates that were 2x too large for the batch it ran. t0004 showed the learning-rate axis is live at 2**18, since a 1/sqrt(2) rescale bought 0.000482 bpb against a repeat noise level near 0.0001. This node moves batch and learning rate together along the square-root scaling line: batch 2**17 with all three main rates at half their original 2**19 values. If the t0003 failure was a rate mismatch, this should now beat champion t0004; if 2**18 is genuinely critical, it will lose even with matched rates, which settles the question.

## t0006: discard

t0006 did not improve the prior champion: val_bpb=0.983237, delta=+0.002090. Hypothesis: Attention head count, not head width, is the under-provisioned quantity in this configuration. With model_dim 512 and HEAD_DIM 128 the model runs only 4 attention heads, well below the 8 or more typical at this width. Because c_q, c_k, c_v and c_proj are all dim-to-dim projections, halving HEAD_DIM to 64 doubles the head count to 8 at exactly the same parameter count, and the FLOP model 12*n_head*head_dim*window is likewise unchanged because n_head*head_dim stays 512. This should buy strictly more attention expressivity for free and lower val_bpb.

## t0007: discard

t0007 did not improve the prior champion: val_bpb=0.987139, delta=+0.005992. Hypothesis: The two elite branches differ only in where they sit on the batch-versus-learning-rate line, and they score within noise of each other, so the line is flat but its correct slope is still unknown. Three points now constrain it at batch 2**17: full 2**19 rates gave 0.986688, half rates gave 0.981248, and the champion at batch 2**18 with 1/sqrt(2) rates gave 0.981147. Because halving the rates recovered almost the entire t0003 penalty, the optimum at a 4x smaller batch may lie below the square-root prescription. This node combines the batch of t0005 with the rate-rescaling mechanism of t0004 pushed from square-root to linear scaling: batch 2**17 with all three rates at 1/4 of their 2**19 values. If linear scaling is the correct law for these orthogonalised Muon updates, this beats both elites.

## t0008: keep

t0008 improved the prior champion: val_bpb=0.980245, delta=-0.000902. Hypothesis: This run is data-limited rather than compute-limited: the loader reports epoch 2, so the 550M training tokens are roughly two passes over the same corpus, while MFU is already 40.6 percent. In that regime the binding constraint is how much of the corpus the model can commit to accessible memory, and the value-embedding path is exactly that mechanism, injecting a per-token learned value directly into the attention value stream. It currently runs on only 4 of the 8 layers. Extending it to every layer should give the model denser token-indexed memory and lower val_bpb, at almost no FLOP cost because value embeddings are gathers rather than matmuls and are excluded from the per-token FLOP model.

## t0009: failed

t0009 ended as failed. Hypothesis: Attention currently reads across document boundaries. The loader packs multiple whole documents into every 2048-token row and starts each row with BOS, so within a 512-token short window a large fraction of queries attend to tokens from a preceding, unrelated document. That cross-document context is pure noise for next-token prediction. Restricting attention to the enclosing document, by deriving cu_seqlens from BOS positions and switching to the flash-attention-3 varlen kernel, should remove that noise and lower val_bpb. This is the first node to change what attention may see rather than how much compute it costs.. Evidence: expected one nonce-bound locked evaluation, found 0; training process exited with code 1; training_seconds is missing from output

## t0010: keep

t0010 improved the prior champion: val_bpb=0.978165, delta=-0.002080. Hypothesis: Unchanged from t0009: attention reads across document boundaries inside each packed 2048-token row, and confining it to the enclosing document should remove that noise and lower val_bpb. t0009 did not test this hypothesis at all; it aborted during compilation because torch._dynamo.mark_dynamic was called from inside GPT.forward, which dynamo traces, raising AssertionError on a forbidden callable. The repair moves that call into eager code so the same masking mechanism can actually run.

## t0011: discard

t0011 did not improve the prior champion: val_bpb=0.978684, delta=+0.000519. Hypothesis: This model needs far less long-range attention than the recipe provides. Two independent measurements point the same way: halving the short window from 1024 to 512 in t0002 gained 0.001882, and document masking in t0010 gained 0.002083 by forbidding attention outside the current document. Both say the useful support of attention is local. The SSSL pattern still spends two of eight layers at the full 2048 context, and those two layers cost 12.6M of the 44.1M attention FLOPs per token. Dropping to a single full-context layer, which the code already forces as the last layer, should convert that into roughly 4 percent more optimizer steps at little quality cost.

## t0012: keep

t0012 improved the prior champion: val_bpb=0.977934, delta=-0.000231. Hypothesis: The t0004 gain was measured on a bundle of three learning rates moved together, and its own risk note recorded that a win would not attribute the gain to any one parameter group. That debt now matters because setup_optimizer puts the value embeddings in the same group as the token embeddings, and t0008 showed the value-embedding path is the most productive mechanism in this recipe, extending it to all eight layers and doubling those parameters to 33.5M. A parameter group that large and that load-bearing may not want the square-root batch rescale that helped the matrix and unembedding groups. Restoring EMBEDDING_LR alone to its original 0.6, while leaving UNEMBEDDING_LR and MATRIX_LR at their rescaled values, decomposes the bundle and tests whether the embedding group was rescaled correctly.

## t0013: keep

t0013 improved the prior champion: val_bpb=0.974229, delta=-0.003705. Hypothesis: Capacity is the last untested major axis, and t0011 supplies the reason to test it now. That node cut about 4 percent of modelled per-token FLOPs and returned only 0.8 percent more steps, so wall-clock per step is strongly sublinear in the FLOP model: fixed overheads such as optimizer work, embedding gathers and the per-step synchronisation dominate. If throughput is sublinear downward it should also be sublinear upward, meaning added depth costs fewer steps than the FLOP ratio predicts. Combined with the data-limited picture, where the epoch counter reads 2 and the one capacity increase already tried (dense value embeddings in t0008) paid off, two extra layers at unchanged width should lower val_bpb.

## t0014: discard

t0014 did not improve the prior champion: val_bpb=0.974842, delta=+0.000612. Hypothesis: The two elite lines are nested rather than divergent, since t0013 was built on t0012 and already carries its embedding rate, so the honest synthesis is to re-ask the t0012 question at the t0013 scale. t0012 established a principle: the embedding group, which holds both wte and every value embedding, wants a faster rate than the square-root batch rescale gave it, and restoring 0.6 beat 0.4243. t0013 then made that group substantially larger, taking value embeddings from 8 tables to 10 and the group to roughly 46M parameters, which is now the largest parameter block in the model. If the principle is real and not just a local fit, the enlarged group should want a faster rate still, so EMBEDDING_LR 0.8 should beat 0.6 at depth 10.

## t0015: discard

t0015 did not improve the prior champion: val_bpb=0.978027, delta=+0.003798. Hypothesis: t0013 proved that capacity is underprovisioned but only measured one way of spending it. It bought 0.003705 bpb for an 18 percent step cost by adding depth at fixed width. The open question is whether width returns more or less per step spent, which decides how all remaining capacity should be allocated. Because HEAD_DIM is 128 and build_model_config rounds model_dim up to a multiple of it, the smallest honest width increment is 512 to 640, giving 5 heads instead of 4. If width and depth were equivalent ways to buy capacity, this should land near t0013 after accounting for its larger step cost; if width is the better axis it should beat 0.974229 outright.

## t0016: keep

t0016 improved the prior champion: val_bpb=0.973422, delta=-0.000807. Hypothesis: Depth is the productive capacity axis and it has not yet been pushed to its turning point. The two capacity nodes now bracket the question: depth 8 to 10 cost 18 percent of steps and gained 0.003705, while width 512 to 640 at depth 10 cost 27 percent of steps and lost 0.003798. Depth therefore buys quality per step spent and width does not, at least at this operating point. Extending depth to 12 at unchanged width should gain again, though by less than the first increment as the step cost compounds. Holding width at 512 also keeps this test clean, since model_dim is what drives the implicit 1/sqrt(model_dim/768) Adam rate scaling that confounded the width node.

## t0017: keep

t0017 improved the prior champion: val_bpb=0.973371, delta=-0.000051. Hypothesis: The depth gains are decaying, 0.003705 for 8 to 10 and 0.000807 for 10 to 12, and there are two competing explanations that imply opposite next moves: genuine diminishing returns to depth, or a matrix learning rate that is increasingly mismatched as depth grows. MATRIX_LR was tuned at depth 8. Muon normalises each matrix update to roughly unit scale, so every layer contributes a similar-magnitude change to the residual stream and a 12-layer stack accumulates about 1.5 times the per-step residual movement that 8 layers did at the same rate. Scaling the matrix rate by sqrt(8/12) should restore the tuned per-step movement. If this wins, the recent depth increments were understated by rate mismatch and depth is worth pushing further; if it loses, depth really is saturating and the axis closes.

## t0018: discard

t0018 did not improve the prior champion: val_bpb=0.974383, delta=+0.001012. Hypothesis: The capacity increases silently changed the data regime, and the regularisation was never revisited. At depth 8 the run consumed 549.7M tokens and the loader reported epoch 2, so every example was seen about twice; at depth 12 it consumes 385.4M tokens and reports epoch 1, a single pass over fresh data. Weight decay earns its keep by suppressing memorisation of repeated examples, and there is no longer any repetition to suppress, so the inherited WEIGHT_DECAY of 0.2 is now mostly a drag on fitting. Halving it to 0.1 should lower val_bpb. This also explains why the value-embedding mechanism, which is memory-like, paid off at depth 8 where data repeated but has not been retested since the regime moved.

## t0019: discard

t0019 did not improve the prior champion: val_bpb=0.973547, delta=+0.000177. Hypothesis: t0018 tested the wrong direction for the right reason and produced a useful gradient. Halving WEIGHT_DECAY from 0.2 to 0.1 cost 0.001012 bpb, which falsifies the reading that this term suppresses memorisation of repeated data: the run is now single-epoch, so if that were its role, removing it should have been free or helpful. The surviving explanation is the one in the t0018 risk note, that cautious weight decay acts as an optimisation constraint on Muon, shrinking coordinates whose update agrees in sign with the parameter and keeping the orthogonalised updates well conditioned. Under that reading more of it should help, and the axis is steep, roughly 0.001 bpb per 0.1 of decay, which is a stronger local gradient than most knobs tested here. Raising it to 0.3 should lower val_bpb.

## t0020: discard

t0020 did not improve the prior champion: val_bpb=0.974838, delta=+0.001468. Hypothesis: The learning-rate schedule is the last untouched cluster in this recipe and it has never been questioned, unlike the rates themselves which have been rescaled three times. The current shape holds the rate constant for the first half of the time budget and then decays linearly to zero over the second half. Annealing is what converts optimisation progress into a low final loss, and at a fixed 300-second budget the anneal is the only phase whose length directly buys final-loss quality. Extending WARMDOWN_RATIO to 0.8 turns the trajectory into a short 20 percent stable phase followed by a long linear decay, close to a pure linear-decay-to-zero schedule, which should reach a lower val_bpb than a schedule that spends half the budget at full rate.

## t0021: discard

t0021 did not improve the prior champion: val_bpb=0.974941, delta=+0.001570. Hypothesis: The two elite lines are nested and differ only in MATRIX_LR, 0.02828 in t0016 against 0.0231 in t0017, and their runs overlap: 0.9734873 and 0.9733571 against 0.9733484 and 0.9733932. Their joint content is therefore not a rate preference but a flatness result, that the Muon matrix rate is insensitive in this neighbourhood at depth 12. That retires the specific concern which motivated t0017, namely that decaying depth returns might be rate mismatch rather than saturation. Combining the shared mechanism of both elites, that depth increments buy quality per step spent, with their joint finding that the rate axis is flat, depth 14 is the test that closes the capacity axis: with rate mismatch excluded, whatever it measures is the genuine marginal value of depth.

## t0022: discard

t0022 did not improve the prior champion: val_bpb=0.975177, delta=+0.001806. Hypothesis: The t0008 result has outlived the conditions that produced it and should be retested rather than inherited. Dense value embeddings won 0.000902 at depth 8, where the run consumed 549.7M tokens and the loader reported epoch 2, so every example was seen roughly twice and token-indexed memory paid off. Depth 12 changed that: the run now consumes 385.4M tokens at epoch 1, a single pass, and the value-embedding block has grown from 8 tables to 12, about 50M parameters, which is the largest block in the model and carries AdamW moments and a gather per layer. Without repetition to memorise, that capacity should be worth less than the steps it costs, so reverting to the original alternating pattern should raise the step count enough to lower val_bpb.

## t0023: discard

t0023 did not improve the prior champion: val_bpb=0.973534, delta=+0.000163. Hypothesis: The value-embedding path is the strongest mechanism in this recipe and it has never had its own learning rate. setup_optimizer already places value_embeds in a separate parameter group, but that group borrows embedding_lr from transformer.wte, so every rate experiment has moved both together. That makes t0014 uninterpretable in a specific way: raising the shared rate from 0.6 to 0.8 lost 0.000613, but the loss cannot be attributed, and wte is the likelier objector since it is a single bf16 table feeding a norm while the value embeddings are twelve tables injected into the attention value stream. The evidence that the mechanism is load-bearing has also grown, from 0.000902 at depth 8 to 0.001806 at depth 12 as measured by t0022. Decoupling the rates and giving the value embeddings 0.9 while wte stays at 0.6 should lower val_bpb.

## t0024: discard

t0024 did not improve the prior champion: val_bpb=0.982521, delta=+0.009151. Hypothesis: This recipe has short skips and embedding skips but no long skips. Every layer already reads the token embedding through x0_lambdas, and every block has its residual path, but nothing carries mid-network representations forward: layer 11 cannot see what layer 0 computed except through eleven successive transformations. U-net style long skips address exactly this, pairing decoder layer i with encoder layer n_layer-1-i through a learned scalar, and they are a known win in the modded-nanogpt lineage this file descends from. At depth 12 the model is deep enough for the gap to matter, and depth is where this campaign found its largest gains, so giving the second half of the stack direct access to the first half should lower val_bpb.

## t0025: discard

t0025 did not improve the prior champion: val_bpb=0.974106, delta=+0.000735. Hypothesis: t0024 did not test the U-net skip mechanism; it tested a mis-parameterised version of it. The evidence is unambiguous: steps were 1465 against the champion 1470 and peak VRAM was 66795.3MB against 66795.6MB, so the skips cost no compute and the entire 0.009150 regression was a quality effect. The cause is the optimizer group I chose. This recipe treats its two scalar groups very differently, giving x0_lambdas the full scalar_lr of 0.5 and resid_lambdas only scalar_lr times 0.01, or 0.005. I gave a zero-initialised parameter the aggressive 0.5 rate, and because Adam normalises its step, a scalar starting at zero moves roughly 0.5 per step and reaches a magnitude of order ten within a hundred steps, which corrupts the residual stream it is added into. Rerunning the identical mechanism at the conservative 0.005 rate used for resid_lambdas should let the skips settle at a useful scale and lower val_bpb.

## t0026: discard

t0026 did not improve the prior champion: val_bpb=0.974129, delta=+0.000758. Hypothesis: The schedule has only been probed on one side. t0020 lengthened the anneal from 0.5 to 0.8 of the time budget and lost 0.001467, which establishes that the constant-rate phase is doing necessary work but says nothing about whether 0.5 is the optimum or merely better than 0.8. The untested direction is a shorter anneal, which spends more of the budget at full rate and compresses the decay. That direction is also the one favoured by everything else this campaign found: the run is now 1470 steps at depth 12, fewer than the 2101 it had at depth 8, so there are fewer optimizer steps in which to make progress before annealing begins, which argues for a later start to the decay. Shortening WARMDOWN_RATIO to 0.35 should lower val_bpb, and together with the 0.8 measurement it brackets the axis either way.

## t0027: discard

t0027 did not improve the prior champion: val_bpb=0.973499, delta=+0.000129. Hypothesis: The AdamW second-moment horizon is mismatched to the gradients it estimates. ADAM_BETAS is (0.8, 0.95), so beta2 averages squared gradients over roughly 20 steps, and it governs four groups of which three are embedding-like: lm_head, wte and the twelve value-embedding tables. Embedding gradients are sparse in a way matrix gradients are not, since a given row is touched only when its token appears in the batch, so its squared-gradient estimate is built from a handful of sporadic observations rather than a dense signal. A 20-step horizon makes that estimate noisy, and a noisy denominator produces effectively random per-row step sizes. This matters more here than in the original recipe because t0022 measured the value-embedding path as the most load-bearing mechanism in the model, worth 0.001806 at depth 12. Lengthening beta2 to 0.98, a roughly 50-step horizon, should stabilise those updates and lower val_bpb.

## t0028: discard

t0028 did not improve the prior champion: val_bpb=0.975732, delta=+0.002361. Hypothesis: The two elite lines jointly say that depth 12 is the right capacity and that the Muon learning rate is flat there, since t0016 at 0.02828 and t0017 at 0.0231 produced overlapping runs. That flatness is the useful part: it means step size is not what limits the Muon path, so if the optimizer for the bulk of the model can still be improved it has to be through its other knob, momentum, which no node has touched. Muon orthogonalises the momentum buffer rather than the raw gradient, so momentum directly sets how much gradient averaging happens before the update direction is computed, and a noisier buffer yields a less reliable orthogonal direction. The schedule ramps momentum from 0.85 to a terminal 0.95 over the first 300 steps and then holds it. Raising that terminal value to 0.98 lengthens the averaging window from roughly 20 to 50 gradients and should produce cleaner update directions and a lower val_bpb.

## t0029: discard

t0029 did not improve the prior champion: val_bpb=0.974423, delta=+0.001052. Hypothesis: Muon momentum is one of the steepest axes measured in this campaign and only one side of it is known. Raising the terminal momentum from 0.95 to 0.98 in t0028 cost 0.002361, which is comparable to the weight-decay gradient and roughly eighteen times the repeat-noise level, so this control has real leverage even though the inherited value may already be near optimal. The same reasoning that worked for weight decay applies: when a single measurement shows a steep response, the opposite direction is worth one node because the optimum need not sit exactly at the inherited default. Lowering the terminal momentum to 0.92, a roughly 12-gradient averaging window instead of 20, tests whether shorter averaging suits a run whose loss surface changes quickly through a 300-second budget.

## t0030: discard

t0030 did not improve the prior champion: val_bpb=0.974557, delta=+0.001186. Hypothesis: The short attention window was set at a different operating point and has never been revisited. t0002 halved it from 1024 to 512 and gained 0.001882, but that measurement was taken at depth 8 with 2099 steps, before document masking existed, before value embeddings went dense, and before capacity moved to depth 12 with 1470 steps. Two things have changed that argue the trade may now reverse. Document masking already forbids attention from crossing into a neighbouring document, so a wider window no longer risks pulling in unrelated text and buys only genuine in-document context. And the earlier win came from converting attention FLOPs into steps, whereas t0011 later showed that FLOP cuts convert into steps very poorly here, only 0.8 percent of steps for a 4 percent FLOP cut, which means the step penalty for widening back to 1024 should be far smaller than the step gain that motivated narrowing it. Restoring the 1024 window should now lower val_bpb.

## t0031: keep

t0031 improved the prior champion: val_bpb=0.973163, delta=-0.000208. Hypothesis: The value-embedding path is the most load-bearing mechanism in this model, worth 0.001806 at depth 12 by the t0022 measurement, yet the gate that controls it sees almost nothing. Each layer computes its gate as 2*sigmoid of a linear map applied to x[..., :32], the first 32 of 512 residual channels, so the decision of how much token-indexed memory to inject is made from six percent of the available context and from an arbitrary slice of it rather than a learned projection of the whole. Widening ve_gate_channels to 128 gives that decision four times the context at negligible cost, since the gate projection is only ve_gate_channels by n_kv_head, which grows from 32x4 to 128x4 weights per layer. If the gate is the bottleneck on a mechanism that is already the strongest in the recipe, this should lower val_bpb.

## t0032: discard

t0032 did not improve the prior champion: val_bpb=0.975306, delta=+0.002143. Hypothesis: Widening the value-embedding gate from 32 to 128 channels in t0031 gained 0.000208 at no step cost, 1470 and 1467 steps against the champion 1470, which is the first improvement in fourteen nodes and says the gate rather than the mechanism was the binding constraint. That node left one question open by construction: it still fed the gate an arbitrary prefix of the residual stream, so a gain could mean either that more context helps or merely that 128 arbitrary channels happen to carry more signal than 32. Setting ve_gate_channels to the full model width of 512 resolves both at once, since the slice then becomes the entire residual stream and the projection is a genuine learned map over all of it rather than over a prefix. If the trend from 32 to 128 continues, this should lower val_bpb again.

## t0033: discard

t0033 did not improve the prior champion: val_bpb=0.976978, delta=+0.003815. Hypothesis: The rotary frequency spectrum is tuned for a context this model never sees. RoPE with base 10000 spreads inverse frequencies so the slowest channels have wavelengths of order 60000 positions, which is appropriate when attention spans thousands of tokens. This model attends over 512 tokens in nine of its twelve layers and at most 2048 in the other three, and document masking cuts many spans shorter still. Across a 512-token window the low-frequency channels rotate by a tiny fraction of a cycle and are therefore nearly constant, so a large part of the head dimension carries almost no positional information. Two independent results in this campaign say the model genuinely wants local attention rather than merely tolerating it: narrowing the window to 512 gained 0.001882 in t0002, and widening it back to 1024 lost 0.001186 in t0030 while costing only 1.2 percent of steps, so it was a quality preference and not a throughput trade. Lowering the base to 1000 matches the spectrum to the span actually used and should give finer positional resolution where the model is looking, lowering val_bpb.

## t0034: keep

t0034 improved the prior champion: val_bpb=0.973097, delta=-0.000065. Hypothesis: t0033 falsified the local-resolution reading of the rotary spectrum and did so with a large effect: lowering the base from 10000 to 1000 cost 0.003815, roughly thirty times the repeat noise. The low-frequency channels that barely rotate inside a 512-token window are therefore not wasted capacity but load-bearing, most plausibly as a slowly varying absolute-position signal that the three full-context layers and the document-masked spans can exploit. If that reading is right the preference is monotonic in base over the range tested, and the untested side is above 10000. Raising the base to 30000 flattens the spectrum further, dedicating even more of the head dimension to slow position signal, and should lower val_bpb if the mechanism that made 1000 so costly continues to pay.

## t0035: discard

t0035 did not improve the prior champion: val_bpb=0.973765, delta=+0.000668. Hypothesis: The two elite lines are nested and differ only by the rotary base, 10000 in t0031 against 30000 in t0034, a difference of 0.000065 that sits below the 0.00013 repeat-noise level. Their joint content is therefore that the rotary axis is inert above 10000 while the value-embedding gate is the live mechanism both share, since widening it from 32 to 128 channels is the only change to have improved the champion in the last twenty nodes. That mechanism has a measured optimum with steep asymmetry: 32 costs 0.000208 relative to 128, and 512 costs 0.002143, so the curve turns sharply somewhere between 128 and 512 and its minimum has only been located to within a factor of four. Testing 256 resolves the bracket at the finest resolution that matters and is the one remaining question on the mechanism the two elites hold in common.

## t0036: discard

t0036 did not improve the prior champion: val_bpb=0.977341, delta=+0.004243. Hypothesis: The output softcap may be limiting the metric directly rather than only stabilising training. The head computes logits, casts to float32, then applies 15*tanh(logits/15), which asymptotically bounds every logit to plus or minus 15 and, more importantly, compresses differences once magnitudes approach that scale. val_bpb is a cross-entropy quantity, so it is exactly the size of the gap between the correct logit and its competitors that sets the achievable loss on confident predictions, and a model at 0.973 bpb makes many highly confident predictions on a small 8192-token vocabulary. If the cap binds on those, it puts a floor under the loss that no amount of better representation can pass. Doubling the cap to 30 keeps the same stabilising shape near the origin, since tanh is linear there, while relaxing the ceiling where it binds, and should lower val_bpb.

## t0037: keep

t0037 improved the prior champion: val_bpb=0.972704, delta=-0.000393. Hypothesis: t0036 falsified the reading that the output softcap puts a floor under the metric: doubling it to 30 cost 0.004244, about thirty times the repeat noise, so the compression is doing optimisation work rather than limiting achievable loss. The most likely mechanism is gradient control into lm_head, since tanh saturation shrinks the gradient on logits that have grown large and prevents the unembedding from chasing overconfident predictions, which matters most in a 300-second run where lm_head starts at a standard deviation of 0.001 and must move quickly. If that is what the cap is for, then more of it should help, exactly as more cautious weight decay helped in the t0018 and t0019 pair. Tightening the cap to 10 tests that and, with the 30 measurement, brackets the axis.

## t0038: discard

t0038 did not improve the prior champion: val_bpb=0.977910, delta=+0.005205. Hypothesis: The output softcap is the first inherited default in this campaign that is not already at its optimum, and the axis is steep and one-sided so far. Loosening it to 30 cost 0.004244 while tightening it to 10 gained 0.000393 against a repeat noise of 0.00013, which supports the gradient-control reading: tanh saturation shrinks the gradient reaching lm_head once logits grow, and in a 300-second run where lm_head starts at standard deviation 0.001 that damping is worth more than the logit range it sacrifices. If more of it continues to help, the optimum lies below 10. Tightening to 7 tests that and, with the 15 and 30 measurements, brackets the axis from both sides.

## t0039: discard

t0039 did not improve the prior champion: val_bpb=0.975856, delta=+0.003151. Hypothesis: The softcap results identify lm_head dynamics as a binding constraint, and the same conclusion implies an untested lever. Across four points the cap has a sharp optimum at 10, with 7 costing 0.005206 and 30 costing 0.004637, and the reading that survives both sides is gradient control: tanh saturation damps the gradient reaching lm_head once logits grow, and that damping is worth real bpb in a 300-second run. If lm_head is being driven too hard, its learning rate is the direct lever on the same quantity, and it is the one parameter group from the t0004 bundle never decomposed. t0012 decomposed the embedding group and found it wanted its original faster rate; the softcap evidence predicts the opposite for the unembedding group, that it wants a slower one. Lowering UNEMBEDDING_LR from 0.002828 to 0.002 tests that prediction directly.

## t0040: discard

t0040 did not improve the prior champion: val_bpb=0.973969, delta=+0.001264. Hypothesis: t0039 falsified the idea that the softcap and the unembedding rate act on the same quantity: lowering UNEMBEDDING_LR from 0.002828 to 0.002 cost 0.003152, roughly twenty-four times the repeat noise, so the cap earns its keep through selective saturation of large logits rather than through any uniform reduction in step size, and lm_head in fact wants to move faster rather than slower. That reopens a question this campaign has already answered once for a neighbouring group. The t0004 bundle divided three rates by sqrt(2) together and won, but t0012 later showed that gain did not belong to the embedding group, which preferred its original 0.6 over the rescaled 0.4243. The unembedding group is the last member of that bundle still carrying the rescale it may never have wanted, and the steep penalty for damping it further now argues the same way. Restoring UNEMBEDDING_LR to its original 0.004 should lower val_bpb.

## t0041: discard

t0041 did not improve the prior champion: val_bpb=0.972811, delta=+0.000107. Hypothesis: The NorMuon variance-reduction horizon is the last untouched control inside the optimizer that trains most of the model. muon_step_fused maintains a second-moment buffer over the orthogonalised update, per row or per column depending on matrix orientation, and rescales by its inverse square root; that buffer uses beta2 of 0.95, a roughly 20-step horizon, and it is a distinct quantity from the Muon momentum of 0.95 that t0028 and t0029 already bracketed. The horizon matters because the quantity being averaged is the squared magnitude of an orthogonalised update, which is far more stationary than a raw gradient, so a longer window should estimate it more precisely at little cost in adaptivity. Raising beta2 to 0.98 should give steadier per-row scaling and lower val_bpb.

## t0042: discard

t0042 did not improve the prior champion: val_bpb=0.975442, delta=+0.002738. Hypothesis: The two elite lines jointly say the optimizer is finished as a source of gains: t0041 moved the last untouched optimizer control, the NorMuon variance horizon, and landed at 0.972811 against the champion 0.972704, a difference below the 0.00013 repeat noise. What remains unexploited is a kernel mechanism the recipe never enables. The single inherited default that proved suboptimal in this entire campaign was the output softcap, where tightening from 15 to 10 gained 0.000393 and both 7 and 30 cost more than 0.004, and its mechanism is bounding logits before a softmax. Attention applies a softmax too, and flash-attention-3 exposes exactly the same control through its softcap argument, which this recipe leaves at the default 0.0, meaning off. Because q and k are RMS-normalised per head, attention logits are bounded near 11.3 at perfect alignment while typical values sit near 1, so a cap of 10 binds only on the sharpest matches and should temper over-confident attention without touching the bulk of the distribution.

## t0043: discard

t0043 did not improve the prior champion: val_bpb=0.973143, delta=+0.000439. Hypothesis: t0042 produced a sharper result than intended and it inverts into a testable prediction. Capping attention logits at 10 cost 0.002738, roughly twenty-one times the repeat noise, even though QK normalisation bounds those logits near 11.3 at perfect alignment so the cap could only bind on the most aligned query-key pairs. The only way so few affected pairs can cost that much is if the sharpest attention matches carry a disproportionate share of the useful signal. If attention sharpness is that valuable, the recipe may be under-sharpening: the kernel uses the conventional 1/sqrt(head_dim) temperature, which is calibrated for unnormalised q and k, whereas this model RMS-normalises both per head and therefore has a tightly bounded logit range that never approaches saturation. Raising the temperature multiplier to 1.5 widens the effective logit range from about 11.3 to 17 and should produce sharper retrieval and a lower val_bpb.

## t0044: failed

t0044 ended as failed. Hypothesis: The attention temperature is a single global constant, but the evidence says the right value is neither uniform nor far from the default. t0042 capped the logit tail and cost 0.002738; t0043 rescaled every logit by 1.5 and cost only 0.000439. Those two results together say the model tolerates a temperature change an order of magnitude better than it tolerates tanh distortion, so the tail matters for its shape rather than its scale, and a global rescale is simply the wrong instrument. What has never been available is per-head control: with QK normalisation fixing the norms of q and k, the only way any individual head can set its own sharpness is through the shared 1/sqrt(head_dim) constant, so all 48 heads across 12 layers are forced to the same temperature. Adding a learned scale per head on q and on k, initialised at 1.0 so the model starts exactly at the champion, lets each head choose its own sharpness and should lower val_bpb.. Evidence: expected one nonce-bound locked evaluation, found 0; training process exited with code 1; training_seconds is missing from output

## t0045: discard

t0045 did not improve the prior champion: val_bpb=0.972759, delta=+0.000054. Hypothesis: Unchanged from t0044: with QK normalisation fixing the norms of q and k, every one of the 48 heads is locked to the same 1/sqrt(head_dim) temperature, and learned per-head scales initialised at 1.0 should let each head choose its own sharpness and lower val_bpb. t0044 never tested that hypothesis because it aborted in the attention kernel with a dtype error, FlashAttention only supports fp16, bf16, and fp8_e4m3. The cause is type promotion rather than the mechanism: under autocast q and k are bf16 while the new scales are fp32 parameters, and elementwise multiply is not an autocast-listed op, so bf16 times fp32 promotes the product to fp32 and fa3 rejects the input.

## t0046: discard

t0046 did not improve the prior champion: val_bpb=0.972982, delta=+0.000278. Hypothesis: Gating is the one control-path addition that has paid in this campaign, and attention output has none. The value-embedding path is gated per head by 2*sigmoid of a linear map on 128 residual channels, and widening exactly that gate from 32 to 128 channels produced the only champion improvement in twenty nodes. Attention output receives no such treatment: every head contributes to the residual stream with a fixed unit weight regardless of context, so a head that is uninformative for a given token cannot be attenuated except through c_proj, which is a fixed linear map shared across all positions. Adding an input-dependent per-head gate on the attention output, built exactly like the value-embedding gate and zero-initialised so it starts neutral at a gate value of 1.0, should let the model suppress unhelpful heads per token and lower val_bpb.

## t0047: discard

t0047 did not improve the prior champion: val_bpb=0.974299, delta=+0.001595. Hypothesis: Capacity has been treated as closed because depth is bracketed at 12, with 10 costing 0.000807 and 14 costing 0.001519, but that bracket only ever varied how many blocks there are and never how capacity is allocated inside one. The allocation matters because the two routes have different prices. Adding two layers to reach depth 14 added about 6.3M matrix parameters but also two more value-embedding tables and two more attention stacks, and cost 13 percent of steps. Raising the MLP expansion from 4 to 5 adds the same 6.3M matrix parameters while adding no attention FLOPs and no value-embedding table, and should cost about 11 percent of steps. If capacity per step spent is what the depth bracket was really measuring, the cheaper route should land better than depth 14 did, and if MLP capacity is genuinely underprovisioned relative to attention it should beat the champion.

## t0048: keep

t0048 improved the prior champion: val_bpb=0.972662, delta=-0.000043. Hypothesis: The schedule endpoint is the last untouched part of the learning-rate trajectory. The warmdown fraction is bracketed at 0.5, with 0.35 costing 0.000758 and 0.8 costing 0.001467, but every one of those runs decayed to exactly zero because FINAL_LR_FRAC is 0.0. That endpoint has a real consequence at this operating point: the run takes about 1465 steps and the decay occupies the final half, so the last hundred or so steps operate below five percent of peak rate and contribute almost nothing to the weights while still consuming their share of the 300-second budget. Holding a floor at ten percent of peak keeps those steps productive. The competing consideration is that annealing to zero is what converts optimisation progress into a low final loss, which is why the recipe does it, so this is a genuine trade rather than free steps.

## t0049: discard

t0049 did not improve the prior champion: val_bpb=0.975154, delta=+0.002492. Hypothesis: The two elite lines differ only in the schedule endpoint, 0.0 in t0037 against 0.1 in t0048, and that difference measured 0.000043, well below the 0.00013 repeat noise, so the floor by itself is inert. The t0048 risk note predicted why: weight decay is driven by the same progress variable but was not given the same floor, so the run now ends at ten percent of peak rate with essentially zero weight decay, a combination the recipe never intended. In muon_step_fused the decay term is lr times wd times the parameter, so effective decay is the product of two independently decaying schedules and vanishes quadratically while the update itself vanishes linearly. Making weight decay track the rate multiplier keeps that ratio fixed across the whole run and is the coherent completion of the floor the two elites bracket.

## t0050: discard

t0050 did not improve the prior champion: val_bpb=0.973772, delta=+0.001110. Hypothesis: t0049 changed two things at once and lost 0.002492: it coupled the weight-decay schedule to the rate multiplier, and in doing so it raised the total amount of decay applied. Those are separable and only the first is the idea worth testing. In muon_step_fused the decay term is lr times wd times the parameter, so the quantity that matters is the integral of their product over the run. For the original schedule that integral is the integral of lrm times 0.2 times (1 - progress), which evaluates to 0.0925 in units of peak rate; for the coupled form it is WEIGHT_DECAY times the integral of lrm squared, which evaluates to 0.685 times WEIGHT_DECAY. Setting WEIGHT_DECAY to 0.135 equates them. This node therefore tests the shape of the coupling at matched total decay: if the coupling is right, holding decay in fixed proportion to update magnitude should beat the champion; if it is not, the axis closes and the t0049 loss is explained entirely by the extra decay rather than by the shape.

## t0051: keep

t0051 improved the prior champion: val_bpb=0.971944, delta=-0.000718. Hypothesis: Batch size is the only major axis still mapped exclusively at the old operating point, and it produced the largest single gain of this campaign. The map was drawn at depth 8 with about 2100 steps, where 2**17 and 2**18 scored 0.981248 and 0.981147 with matched rates, a flat ridge. The champion now runs depth 12 with 1465 steps, roughly thirty percent fewer optimizer steps for a model with fifty percent more layers, which is exactly the regime where trading batch width for step count should be worth more than it was before. The square-root rate law that t0003, t0005 and t0007 established at a 4x batch range is the correct accompaniment, so this halves the batch and divides all three rates by sqrt(2). If the ridge is still flat the axis closes at the current operating point; if step count is now scarcer than gradient quality, this wins.

## t0052: discard

t0052 did not improve the prior champion: val_bpb=0.975888, delta=+0.003944. Hypothesis: The batch axis has genuinely moved with the operating point and is not yet exhausted. At depth 8 with about 2100 steps, halving the batch from 2**18 to 2**17 with matched rates was flat, 0.981248 against 0.981147. At depth 12 with 1465 steps the same halving now gains 0.000718, roughly five times the repeat noise, and takes the step count from 1465 to 2838. The mechanism is that a deeper model with fewer optimizer steps is step-starved rather than gradient-starved, so converting batch width into steps pays where it previously did not. If that reading is right the axis should still have room, since 2838 steps at depth 12 is only slightly more than the 2100 the shallower model had. Halving again to 2**16 with all three rates divided by sqrt(2) once more tests whether the gain continues or whether the critical batch size has now been crossed.

## t0053: keep

t0053 improved the prior champion: val_bpb=0.971630, delta=-0.000314. Hypothesis: The winning batch move was a bundle of four constants and one of them has already been shown, at a different batch size, to prefer the opposite treatment. t0051 halved the batch to 2**17 and divided all three rates by sqrt(2), gaining 0.000718. But t0012 established that the embedding group, which drives both wte and all twelve value-embedding tables, prefers its unscaled 0.6 over a sqrt(2)-reduced 0.4243, and that finding was itself the decomposition of an earlier bundle that had lowered it. The same logic applies again: the t0051 gain is most plausibly attributable to the batch change together with the matrix and unembedding rescaling, with the embedding group dragged along rather than helped. Restoring EMBEDDING_LR to 0.6 at the new batch should lower val_bpb further.

## t0054: discard

t0054 did not improve the prior champion: val_bpb=0.971690, delta=+0.000060. Hypothesis: A sharper rule may govern the rates than the uniform square-root law currently in the champion. Two decompositions now point the same way for the embedding group: at batch 2**18 it preferred 0.6 over 0.4243 by 0.000208, and at batch 2**17 it again preferred 0.6 over 0.4243 by 0.000314, so its optimum did not move when the batch halved. The unembedding group has a precisely measured optimum at 2**18 as well, since t0039 and t0040 bracketed it at 0.002828 with 0.002 costing 0.003152 and 0.004 costing 0.001265, and that group is unusually sensitive. The champion currently runs it at 0.002, which is exactly the value measured to be badly wrong one batch size up. If the Adam groups have batch-invariant optima and only the orthogonalised Muon updates follow square-root scaling, restoring 0.002828 should lower val_bpb and would establish a much cleaner rule than the uniform one.

## t0055: discard

t0055 did not improve the prior champion: val_bpb=0.974787, delta=+0.003157. Hypothesis: One member of the t0051 bundle is still undecomposed, and it is the only one the square-root law was ever theoretically motivated for. The decompositions so far give a mixed picture: the embedding group prefers its unscaled 0.6 at both 2**18 and 2**17, twice confirmed, while the unembedding group turned out flat at 2**17, moving only 0.00006 between 0.002 and 0.002828 despite a 0.003152 penalty for the same move one batch size up. That leaves the Muon group, which is the one whose updates are orthogonalised to roughly unit scale and therefore the one where a noise-driven rate law should apply most cleanly. If square-root scaling is real and specific to Muon, then reverting MATRIX_LR to its unscaled 0.0231 should regress; if it does not regress, the t0051 gain came from step count alone and no group actually needed rescaling, which would simplify the recipe and contradict the reading this campaign has carried since t0004.

## t0056: discard

t0056 did not improve the prior champion: val_bpb=0.972701, delta=+0.001071. Hypothesis: Every schedule in this recipe is indexed by elapsed training time except one, and that one has silently drifted as the campaign changed the step count. get_lr_multiplier and get_weight_decay both take progress, which is total_training_time over TIME_BUDGET, so their shape is invariant to how many steps fit in 300 seconds. get_muon_momentum instead takes the raw step index and ramps momentum from 0.85 to 0.95 over the first 300 steps. At the original 1019 steps that warmup covered 29 percent of training; the champion now runs 2838 steps, so it covers 10.6 percent, and the model spends a far larger share of the run at high momentum than the recipe was tuned for. That matters because t0028 and t0029 showed momentum is one of the steepest axes here, with 0.98 costing 0.002361 and 0.92 costing 0.001053. Rescaling the warmup to 850 steps restores the original 30 percent fraction and should lower val_bpb.

## t0057: discard

t0057 did not improve the prior champion: val_bpb=0.971680, delta=+0.000049. Hypothesis: t0056 tested whether the step-indexed momentum warmup should hold a constant fraction of the run and answered no: stretching it from 300 to 850 steps, restoring the 30 percent fraction the recipe had at 1019 steps, cost 0.001071. The reading that survives is that reaching high momentum early is actively valuable rather than that the warmup should scale with run length, which also fits the terminal-value bracket where 0.92 cost 0.001053 and 0.98 cost 0.002361, both worse than 0.95. If early momentum is what matters, the warmup should be shorter than the inherited 300 steps rather than longer, and at 2838 steps a 150-step ramp reaches terminal momentum within the first 5 percent of training. Halving it should lower val_bpb, and together with the 850 measurement it brackets the axis.

## t0058: timeout

t0058 ended as timeout. Hypothesis: Every axis this campaign has probed is now closed except kernel selection, and the model runs at about 41 percent MFU with unusually small GEMMs: a 512-wide residual stream, a 2048-wide MLP and an 8192-row vocabulary. Those shapes are exactly where the default inductor heuristics are least likely to pick the best tile and stage configuration, and where autotuning has the most to recover. The decisive structural fact is that compilation happens lazily on the first forward pass, and the training loop only accumulates total_training_time for steps beyond the tenth, so autotuning cost lands in wall clock rather than in the 300-second budget. If autotuning finds better kernels, every step after the tenth is faster and the step count rises at identical quality per step, which converts directly into lower val_bpb.. Evidence: expected one nonce-bound locked evaluation, found 0; process exceeded the wall-clock safety timeout; training process exited with code 143

## t0059: discard

t0059 did not improve the prior champion: val_bpb=0.971716, delta=+0.000086. Hypothesis: Unchanged from t0058: autotuned kernel selection should raise the step count at identical quality per step, since this model runs near 41 percent MFU on small GEMMs where the default inductor heuristics are weakest. t0058 did not test that hypothesis, it aborted at 661 seconds against the 660-second process timeout while still benchmarking Triton matmul candidates past number 1750. The diagnosis is that mode max-autotune-no-cudagraphs enables far more than GEMM tuning: it also autotunes pointwise kernels, benchmarks epilogue fusions and runs coordinate-descent tuning, and the combined first-compile search does not fit the wall budget. Enabling only max_autotune_gemm keeps the part of the search aimed at where this model actually spends its time and drops the rest.

## t0060: discard

t0060 did not improve the prior champion: val_bpb=0.973431, delta=+0.001801. Hypothesis: The depth bracket was measured at a batch size the champion no longer uses, and one specific mechanism argues the verdict should be re-checked rather than inherited. Depth 14 was rejected at batch 2**18 where it ran only 1278 optimizer steps, against 1465 for depth 12, and deeper residual stacks generally need more steps to reach a given loss because signal must propagate through more layers before the later ones receive useful gradient. At batch 2**17 the champion runs 2838 steps, so depth 14 should now get roughly 2426, nearly double what it had when it lost. The token count is essentially unchanged between the two operating points, 372M for the champion here against 384M at the old batch, so this is not a data-quantity question: it isolates whether the depth-14 loss of 0.001519 was about capacity being unaffordable or about there being too few steps to train it.

## t0061: discard

t0061 did not improve the prior champion: val_bpb=0.973505, delta=+0.001875. Hypothesis: Learning-rate warmup is the last schedule parameter never tested, and the operating point has moved into the regime where it should matter most. The recipe ships with WARMUP_RATIO at 0.0, so every group starts at full rate on step one, which is defensible at a large batch where the first gradients are already well averaged. The champion now trains at batch 2**17, a quarter of the 2**19 the recipe was written for, so early gradients are roughly twice as noisy per step as they were at the tuned setting, and the parameters least protected are the ones that start near zero: lm_head at standard deviation 0.001 and the zero-initialised output projections in every block. A short warmup over two percent of the budget, about 57 steps, lets those gradients average before full-rate updates begin and should lower val_bpb.

## t0062: discard

t0062 did not improve the prior champion: val_bpb=0.974451, delta=+0.002821. Hypothesis: SCALAR_LR is the last learning rate in the model never measured, and it sets a step size that looks large relative to the parameters it moves. It drives two groups through setup_optimizer: x0_lambdas at the full 0.5 and resid_lambdas at 0.005. The x0 group is the striking one. Those twelve scalars are initialised at 0.1 and control how much of the normalised token embedding is re-injected at each layer, and because Adam normalises its update to roughly the learning rate for a consistently signed gradient, a rate of 0.5 permits a single step to move a parameter by five times its own initial value. Once such a parameter reaches its optimum the same rate becomes steady-state jitter of comparable magnitude, so the quantity being learned may be permanently noisy rather than converged. Halving SCALAR_LR to 0.25 should reduce that jitter while leaving ample room to travel over 2838 steps, and should lower val_bpb.

## t0063: discard

t0063 did not improve the prior champion: val_bpb=0.972504, delta=+0.000873. Hypothesis: Both elite lines converge on one principle, that this recipe wants its per-layer control quantities to move fast and early, and the scalar rate is the one place that principle has only been tested downward. t0057 and its parent established the momentum half: stretching the Muon momentum warmup to 850 steps cost 0.001071 while the inherited 300 held, so reaching terminal momentum early is actively valuable. t0062 then established the scalar half from the other side: halving SCALAR_LR to 0.25 cost 0.002821, one of the larger single-knob regressions in this campaign, so the per-layer scalars are hurt by moving slower. The synthesis is that the axis is steep and has been probed on only one side, exactly the situation where t0036 and t0037 found this campaign its single suboptimal default. Doubling SCALAR_LR to 1.0 tests whether 0.5 is the optimum or merely better than 0.25.

## t0064: discard

t0064 did not improve the prior champion: val_bpb=0.978616, delta=+0.006986. Hypothesis: The head-granularity axis was measured on one side only and the untested side is free. t0006 halved HEAD_DIM to 64, giving 8 heads instead of 4, and lost 0.001989 with the step count almost unchanged at 3957 against 4078, so the loss was quality rather than throughput and wider heads beat narrower ones at this width. The opposite direction has never been tried. Because build_model_config derives n_head as model_dim over HEAD_DIM, setting HEAD_DIM to 256 gives 2 heads of width 256 while n_head times head_dim stays 512, so every projection keeps its shape, the parameter count is unchanged, and the FLOP model 12*n_head*head_dim*window is identical. The only thing that changes is how the 512 channels are partitioned into attention subspaces. If the t0006 trend continues, fewer and wider heads should lower val_bpb at no cost.

## t0065: keep

t0065 improved the prior champion: val_bpb=0.971285, delta=-0.000345. Hypothesis: The batch halving silently increased the total weight decay this model receives, and the constant was never rescaled to compensate. In muon_step_fused the decay term is applied per optimizer step as lr times wd times the parameter, so cumulative decay across a run is the step count times the average of that product. Moving from batch 2**18 to 2**17 raised the step count from 1465 to 2838, a factor of 1.937, while the square-root rate law lowered MATRIX_LR from 0.0231 to 0.0163, a factor of 0.707. The net effect is 1.37 times more cumulative decay than the configuration in which WEIGHT_DECAY was bracketed at 0.2 against 0.1 and 0.3. That bracket was also asymmetric, with 0.1 costing 0.001012 and 0.3 costing only 0.000176, so the champion now sits on the expensive side of a curve it was tuned on. Dividing the constant by 1.37 to 0.146 restores the total decay that measured best and should lower val_bpb.

## t0066: discard

t0066 did not improve the prior champion: val_bpb=0.971465, delta=+0.000179. Hypothesis: Rescaling weight decay to hold cumulative decay constant across the batch change gained 0.000345, which confirms that the relevant quantity is the integral of lr times wd over the run rather than its per-step magnitude, since the correction was derived from exactly that assumption. What it does not establish is that the previously optimal cumulative amount is still optimal. The batch halving changed more than the arithmetic of decay: gradients per step are noisier, there are 1.94 times as many updates, and the model now sees each parameter shrunk more often but by less each time. The old bracket found 0.2 optimal against 0.1 and 0.3 at the previous operating point, and the matched value 0.146 is the translation of that optimum, not a fresh measurement. Testing 0.11, roughly a further factor of 1.33 down, asks whether the optimum in cumulative terms has itself moved lower now that decay is applied more frequently and more gently.

## t0067: discard

t0067 did not improve the prior champion: val_bpb=0.971522, delta=+0.000237. Hypothesis: Adam beta1 has never been tested in this campaign and it is set unusually low. The recipe uses betas of 0.8 and 0.95, and while t0027 probed beta2 and measured a null, the first moment was left alone. A beta1 of 0.8 averages gradients over roughly 5 steps against the conventional 0.9 which averages over about 10, and that choice is defensible at the batch the recipe was written for, 2**19, where each gradient is already well averaged across many tokens. The champion now trains at 2**17, a quarter of that, so each per-step gradient carries roughly twice the noise it did at the tuned setting, and the natural compensation for noisier gradients is more temporal averaging rather than less. This mirrors the correction that just paid off at t0065, where a constant tuned at the old batch had to be rescaled for the new one. Raising beta1 to 0.9 should lower val_bpb.

## t0068: discard

t0068 did not improve the prior champion: val_bpb=0.972084, delta=+0.000799. Hypothesis: Two independent momentum-like controls in this recipe have now each rejected more temporal averaging, and this node tests whether that is a principle or a coincidence. Raising the Muon terminal momentum from 0.95 to 0.98 cost 0.002361 while lowering it to 0.92 cost only 0.001053, an asymmetry favouring less averaging, and raising Adam beta1 from 0.8 to 0.9 just cost 0.000237. A coherent explanation is that within a 300-second budget the loss surface moves quickly relative to any averaging window, so stale gradient directions are worth less here than responsiveness, and the recipe compensates by keeping every averaging horizon short. If that is right, beta1 should have room below its inherited 0.8, which is already low by convention. Lowering it to 0.7, roughly a 3-step horizon, tests the principle where it is cheapest to test and brackets the axis either way.

## t0069: keep

t0069 improved the prior champion: val_bpb=0.971071, delta=-0.000214. Hypothesis: The output softcap is the steepest axis in this campaign and it was bracketed at a batch size the champion no longer uses, and a second result suggests its optimum has since moved. The bracket at batch 2**18 gave 7 costing 0.005206, 10 best, 15 costing 0.000393 and 30 costing 0.004637, and the mechanism that survived both sides was damping of the gradient into lm_head. Independent evidence says lm_head is less sensitive at the current batch: the unembedding learning rate was sharply peaked at 2**18, where moving it 30 percent cost 0.003152, but measured flat at 2**17 in t0054, moving only 0.00006 across the same span. If lm_head is genuinely less binding now, it needs less damping, so the softcap optimum should have moved upward from 10 toward the looser end where 15 sat only 0.000393 away. Testing 13 probes that shift on the gentle side of the curve rather than the steep one.

## t0070: discard

t0070 did not improve the prior champion: val_bpb=0.971571, delta=+0.000500. Hypothesis: The two elite lines are both re-measurements of constants that the batch change displaced, and together they establish a pattern worth acting on systematically. t0065 rescaled weight decay by a derivable law, matching the cumulative product of rate and decay, and gained 0.000345; t0069 raised the output softcap from 10 to 13 by inference from lm_head becoming less sensitive, and gained 0.000214. Both were tuned at batch 2**18 and both had moved. Several constants remain that were bracketed at that batch and never re-checked, and the one with the largest measured leverage is the warmdown fraction, where 0.35 cost 0.000758 and 0.8 cost 0.001467. There is also a specific reason to expect its optimum to have moved down rather than stayed: the anneal now spans about 1420 steps instead of 730, so it has twice the steps to settle the model, while t0061 showed this recipe is unusually costly to deprive of full-rate steps, losing 0.001875 for giving up just two percent of the budget to warmup. Shortening the warmdown to 0.4 buys back ten percent of the budget at full rate while still leaving a longer anneal in absolute steps than the champion had at the previous batch.

## t0071: discard

t0071 did not improve the prior champion: val_bpb=0.976498, delta=+0.005427. Hypothesis: Grouped-query attention is the last untested architectural axis, and my reason for skipping it so far has been inference rather than measurement, which this campaign has repeatedly shown to be unreliable. Halving n_kv_head from 4 to 2 shrinks c_k and c_v from full width to half, saving about 3.1M matrix parameters and roughly 8 percent of matrix FLOPs, which should buy a few percent more steps. The cost is that kv_dim is n_kv_head times head_dim, so it falls from 512 to 256 and every value-embedding table halves with it, taking that block from about 50M parameters to 25M. t0022 measured that halving the number of value-embedding tables costs 0.001806, so if width behaves like count this should lose by a similar margin and the step gain will not cover it. The node is worth spending precisely because that transfer from count to width is an assumption: value embeddings may be redundant across their width in a way they are not across layers, in which case GQA is close to free and this wins.

## t0072: discard

t0072 did not improve the prior champion: val_bpb=0.971081, delta=+0.000010. Hypothesis: The embedding skip is a per-layer quantity whose total was silently increased when depth changed, in exactly the way weight decay was when the batch changed. In GPT.forward each layer computes x as resid_lambdas[i] times x plus x0_lambdas[i] times x0, so the normalised token embedding is re-injected once per layer and the total injection across the stack is the sum of those scalars. At depth 8, where the recipe was tuned, that sum is 8 times 0.1 or 0.8; at the champion depth of 12 it is 1.2, half again as much, and no node has ever revisited the initialisation. The same cumulative reasoning corrected weight decay at t0065 for a gain of 0.000345, and the correction there was validated by t0066 finding the translated value optimal. Scaling the initialisation by 8/12 to 0.067 restores the tuned total and should lower val_bpb.

## t0073: keep

t0073 improved the prior champion: val_bpb=0.970589, delta=-0.000482. Hypothesis: Full-context layers are worth more than their cost, and the pattern that decides how many of them exist has never been tested at the current depth. The only measurement is t0011, which at depth 8 replaced SSSL with S, cutting the two full-context layers to one, and lost 0.000519 while gaining just 0.8 percent in steps: removing a long layer was a bad trade. That says long layers carry value the FLOP model understates, but it was measured with 8 layers and the champion now has 12, where SSSL places long layers at indices 3, 7 and 11. Moving to SSL places them at 2, 5, 8 and 11, adding a fourth. If the t0011 direction holds, adding a long layer should pay for its roughly 3 percent FLOP cost. This also probes something the campaign has otherwise found consistently: attention span reductions kept helping until they did not, and this is the opposite end of that same axis at a depth where it was never examined.

## t0074: discard

t0074 did not improve the prior champion: val_bpb=0.970963, delta=+0.000373. Hypothesis: The count of full-context layers is a live axis and it moved in the direction I predicted against. Going from SSSL to SSL at depth 12, three long layers to four, gained 0.000482 while costing only 5 steps out of 2841. That is a correction to the reading I had been carrying: this model prefers a narrow short-window span, which t0002 and t0030 both established, but it does not prefer fewer global layers, and those are separate questions about attention that the campaign had been treating as one. The remaining question is where the count saturates. Moving to SL places long layers at indices 1, 3, 5, 7, 9 and 11, giving six full-context layers against six short ones, and roughly doubles the share of attention spent at full context. If global layers keep paying at their measured rate this should gain again; if four was near the optimum this brackets the axis.

## t0075: discard

t0075 did not improve the prior champion: val_bpb=0.970968, delta=+0.000379. Hypothesis: The number of full-context layers is now bracketed and its optimum lies between the two tested values. At depth 12, three long layers scored 0.971071, four scored 0.970589 and six scored 0.970963, so the curve turns somewhere between four and six and the untested integer is five. The saturation has a mechanism behind it: document masking prevents attention from crossing document boundaries in every layer, so a full-context layer sees more than the 512-token short window only when the enclosing document is longer than that, and once enough global layers exist to route document-level information the rest buy little while still costing FLOPs. Five long layers tests whether the optimum sits just above four or exactly at it. A pattern of SSLSL cycled over twelve layers, with the last forced to full context, places long layers at indices 2, 4, 7, 9 and 11.

## t0076: keep

t0076 improved the prior champion: val_bpb=0.970135, delta=-0.000454. Hypothesis: The short-window span and the number of global layers are coupled, and the span was fixed before the count changed. The 512-token span was chosen in t0002 at depth 8 and re-checked in t0030 at depth 12, but both were measured with the SSSL pattern, which places only three full-context layers. The champion now runs SSL with four, and t0073 showed that fourth global layer is worth 0.000482, which means more document-level routing capacity now exists than when the span was set. If the short layers were carrying medium-range context that the global layers can now absorb, the span can be narrowed further and the freed FLOPs converted into steps. Halving it to sequence_len // 8, or 256 tokens, removes about 12.6M per-token FLOPs from a total near 326M. This is also the one span setting never tested: 1024 and 512 have both been measured, 256 has not.

## t0077: discard

t0077 did not improve the prior champion: val_bpb=0.970591, delta=+0.000456. Hypothesis: The two elite lines are the two halves of a coupling this campaign discovered, and closing the loop is the synthesis they license. t0073 added a fourth full-context layer and gained 0.000482; t0076 then narrowed the short span from 512 to 256 and gained 0.000452, on the reasoning that the extra global layer had absorbed the medium-range context the short layers were carrying. The step counts confirm the mechanism, rising from 2836 to 2874. If the axes are genuinely coupled then the implication runs the other way too: a fifth global layer was tested at the 512 span in t0075 and lost 0.000379, but the short layers now cover half as much ground, so the document-level routing that the fifth layer would provide is worth more than it was. Testing SSLSL at the 256 span asks whether the earlier rejection was conditional on the span rather than final.

## t0078: discard

t0078 did not improve the prior champion: val_bpb=0.972754, delta=+0.002620. Hypothesis: The short-window span is a live axis with a measured gradient and its next point is untested. Narrowing from 1024 to 512 gained 0.001882 at depth 8 and narrowing from 512 to 256 gained 0.000452 at the current configuration, with the step count confirming the mechanism by rising from 2836 to 2874. Halving again to 128 tokens is the next point on that curve. The returns are clearly diminishing, 0.0019 then 0.0005, so a smaller gain or the first regression is expected, and either outcome brackets the axis. t0077 also just showed that the span effect and the global-layer count are close to additive rather than coupled, since narrowing the span was worth 0.000452 at four long layers and 0.000377 at five, which means this can be read as a clean single-axis measurement rather than an interaction.

## t0079: discard

t0079 did not improve the prior champion: val_bpb=0.970423, delta=+0.000288. Hypothesis: The short-window curve is now bracketed on both sides and its shape says the minimum is not exactly at the point measured. At the current configuration 512 costs 0.000454, 256 is the champion and 128 costs 0.002619, so the penalty for halving below the champion is roughly six times the penalty for doubling above it. A curve that steep on one side and shallow on the other has its true minimum displaced toward the shallow side, which here means above 256 rather than below. The span has only ever been sampled at powers of two because the expression is a divisor of the 2048 long window, but nothing requires that: setting it to long_window times 3 over 16 gives 384, the midpoint in log space between 256 and 512, and tests whether the optimum sits between the two sampled points.

## t0080: discard

t0080 did not improve the prior champion: val_bpb=0.970468, delta=+0.000333. Hypothesis: Every short layer is forced to the same span, and the span results suggest that is the wrong constraint rather than the wrong value. Uniform 256 beats uniform 512 by 0.000454 and uniform 128 by 0.002619, so the optimum is sharp, but it is an optimum over a single number applied to eight layers at very different depths. Early layers build local features from raw token embeddings and plausibly need little context, while later short layers operate on representations that have already been mixed by the global layers below them and may want more. A graded schedule tests that directly: the eight short layers take spans rising linearly from 128 to 384 instead of all sitting at 256. The mean span is held at exactly 256 by construction, so this isolates the distribution of context across depth from the amount of it, and any difference is attributable to grading rather than to spending more or less attention overall.

## t0081: discard

t0081 did not improve the prior champion: val_bpb=0.970276, delta=+0.000141. Hypothesis: t0080 tested whether short-layer receptive fields should grow with depth and found they should not: an ascending schedule from 128 to 384 at a held mean cost 0.000333 against the uniform 256 champion. That result cannot distinguish between grading being useless and grading being backwards, which is the ambiguity its own risk note recorded, and the descending direction is the cheap way to resolve it. There is a real argument for descending: early layers work on raw token embeddings and must disambiguate words from local context, which is exactly when a wider window helps, whereas later short layers sit above three full-context layers that have already routed document-level information and may need only a narrow local view. This node reverses the schedule to run from 384 down to 128, holding the mean span at 255.6 as before so the attention FLOP total is unchanged and only the allocation across depth differs.

## t0082: discard

t0082 did not improve the prior champion: val_bpb=0.970253, delta=+0.000118. Hypothesis: The NorMuon variance horizon was measured at a different batch and the argument for re-testing it is the one that has paid twice in this campaign. t0041 raised beta2 from 0.95 to 0.98 at batch 2**18 and measured 0.000107, below the repeat noise, so the axis was closed as a null. The champion now trains at 2**17, where each per-step gradient carries roughly twice the noise, and this buffer estimates the squared magnitude of the orthogonalised update per row in order to rescale it. A noisier input to that estimate is exactly the condition under which a longer averaging window should help, and it is the same reasoning that made the weight-decay rescale at t0065 and the softcap shift at t0069 pay off after the batch moved. Raising beta2 to 0.98 lengthens the horizon from roughly 20 steps to 50.

## t0083: discard

t0083 did not improve the prior champion: val_bpb=0.970621, delta=+0.000486. Hypothesis: Depth is the only axis in this campaign measured at even integers alone, and the odd value between the champion and the rejected one has never been tried. Depth 12 is the champion, 10 cost 0.000807 and 14 cost 0.001519 at batch 2**18 and 0.001801 again at 2**17, so the curve is established but only at a spacing of two. Two changes since those measurements make 13 worth one node rather than an obvious loser: the short attention span narrowed from 512 to 256, cutting per-token attention cost by roughly 3 percent, and the batch halved so the run now takes 2874 steps rather than 1465, which is the regime where a deeper stack has more updates to converge in. Verified arithmetic: ASPECT_RATIO 39 with depth 13 gives a base of 507, which build_model_config rounds up to a model_dim of exactly 512 with 4 heads of width 128, so width is held and only depth moves.

## t0084: discard

t0084 did not improve the prior champion: val_bpb=0.972235, delta=+0.002100. Hypothesis: The within-block allocation of capacity has been measured in one direction only. t0047 raised the MLP expansion from 4 to 5 and lost 0.001595, almost exactly matching the 0.001519 that adding two layers cost, which established that capacity is saturated regardless of how it is bought. The downward direction was never tested and is not symmetric with the upward one: reducing the MLP frees FLOPs that convert into steps, and steps have proven valuable at this operating point, since halving the batch to gain them was worth 0.000718 and narrowing the attention span to gain them was worth 0.000452. Reducing the expansion from 4 to 3 removes about 17 percent of matrix FLOPs and should raise the step count from 2874 toward roughly 3300. The two elite lines agree that the optimizer is exhausted as a source of gains, since t0082 measured the last untested Muon control as null, so what remains is the compute budget itself and how it is spent.

## t0085: discard

t0085 did not improve the prior champion: val_bpb=0.970922, delta=+0.000787. Hypothesis: The unembedding initialisation is the last untested quantity on the parameter this campaign found most sensitive. Three separate results say lm_head dynamics dominate the early run: the output softcap is the steepest axis measured, where 7 cost 0.005206 and 30 cost 0.004637 around an optimum that later shifted from 10 to 13, and its surviving mechanism is damping the gradient into lm_head; and the unembedding rate was sharply peaked at the old batch, where a 30 percent move cost 0.003152. The initialisation sits directly upstream of both. It is currently normal with standard deviation 0.001, which is near zero but not zero, so the model begins with small random logit structure that carries no information and must be unlearned before useful structure replaces it. Every other output projection in this architecture is initialised to exact zeros, including attn.c_proj and mlp.c_proj, so lm_head is the one inconsistency. Zeroing it starts the model at exactly uniform predictions and should lower val_bpb.

## t0086: discard

t0086 did not improve the prior champion: val_bpb=0.971715, delta=+0.001580. Hypothesis: The unembedding initialisation turned out to be a live axis rather than a formality, and only one side of it has been measured. t0085 replaced the standard deviation of 0.001 with exact zeros, expecting a null or a small gain on the argument that random logit structure carries no information and must be unlearned, and it cost 0.000787. That refutes the argument: the small random init is doing something useful, most plausibly breaking the symmetry among vocabulary rows so that the first updates differentiate them rather than moving every row identically. If asymmetry at initialisation is what matters, more of it should help, and the inherited 0.001 is not obviously the right magnitude since it produces logits of order 0.02 against a softcap of 13. Raising it tenfold to 0.01 tests that and, with the zero measurement, brackets the axis.

## t0087: keep

t0087 improved the prior champion: val_bpb=0.969277, delta=-0.000858. Hypothesis: The value embeddings are initialised with a scale meant for a different kind of parameter, and the mismatch is large and quantifiable. init_weights computes s as sqrt(3) times n_embd to the minus one half, which is a fan-in scale appropriate for the projections c_q, c_k, c_v and c_fc, each of which sums 512 inputs. It then applies the same s to the value-embedding tables, which are embeddings with fan-in one: a single row is selected per token. The consequence is that a value embedding starts with element RMS s over sqrt(3), about 0.044, while the stream it is added to, v equals c_v of x, has element RMS 1.0, so the path begins 22.6 times weaker than what it modifies. That transformer.wte is initialised with normal of standard deviation 1.0 rather than with s shows the file already distinguishes embeddings from projections elsewhere. This matters because the value-embedding path is the most load-bearing mechanism measured in this campaign: extending it to every layer gained 0.001806 and halving its width cost 0.005427. Initialising it to match the value stream should let it contribute from the first step instead of having to grow into relevance within 300 seconds.

## t0088: discard

t0088 did not improve the prior champion: val_bpb=0.969342, delta=+0.000065. Hypothesis: Correcting the value-embedding initialisation was the largest single gain since the depth work, and its optimum is not yet bracketed. t0087 replaced the fan-in scale s, which gave element RMS 0.044, with plus or minus sqrt(3), giving RMS 1.0 to match the v stream, and gained 0.000858 against a repeat noise of about 0.00013. That confirms the diagnosis, that a scale meant for a 512-input projection was wrongly applied to a fan-in-one embedding table, but matching v exactly was a principled target rather than a measured optimum. The mechanism argues the useful scale could be larger: the gate multiplies the embedding by 2*sigmoid before adding, so at the zero-initialised gate the factor is exactly 1.0, and the value stream carries v plus ve; if the model wants the memory path to dominate the projected value early, an embedding larger than v would let it. Doubling to RMS 2.0 tests whether the optimum sits at parity or beyond it.

## t0089: keep

t0089 improved the prior champion: val_bpb=0.968389, delta=-0.000888. Hypothesis: Correcting the value-embedding initialisation changed what learning rate that path should want, and the earlier measurement of its rate was taken under the broken init. t0087 rescaled the tables from element RMS 0.044 to 1.0, matching the v stream, and gained 0.000858; t0088 then showed the axis is flat above parity, so the scale is right. Under the old init the tables began 22.6 times too small and had to grow into relevance inside 300 seconds, which is a condition that rewards a fast rate. That is the regime in which t0023 tested a decoupled value-embedding rate of 0.9 and found it slightly worse than the shared 0.6, and in which every other embedding-rate result was collected. Now that the path starts at the scale it needs, the rate no longer has to fund growth and can instead be tuned for stability, so a lower rate should help. This node decouples the two groups again and halves the value-embedding rate to 0.3 while wte keeps 0.6.

## t0090: discard

t0090 did not improve the prior champion: val_bpb=0.968528, delta=+0.000139. Hypothesis: The value-embedding rate is a live axis with a large measured gradient and only one side explored. Halving it from the shared 0.6 to a decoupled 0.3 gained 0.000888, one of the largest single gains in this campaign and about seven times the repeat noise. The mechanism behind it is specific: t0087 fixed an initialisation that had these tables starting 22.6 times smaller than the v stream they feed, and a path that no longer has to grow into relevance does not need a rate sized for growth. That argument does not stop at 0.3. If the rate was funding growth that the init now supplies, the correct rate is whatever suits a table already at the right scale, and nothing measured so far locates it. Halving again to 0.15 tests whether the gradient continues or whether 0.3 is the turning point, and either result brackets the axis against the 0.6 measurement.

## t0091: discard

t0091 did not improve the prior champion: val_bpb=0.968898, delta=+0.000509. Hypothesis: Both elite lines are consequences of one repair, and they point at a third quantity that was measured under the same defect. t0087 corrected a value-embedding initialisation that started the tables 22.6 times weaker than the v stream, and t0089 then showed that the learning rate for that group had been sized to fund the growth the init now supplies, gaining 0.000888 by halving it; t0090 bracketed that rate at 0.3. The gate is the remaining control on the same path and it was tuned in exactly the broken regime. Its width was bracketed across four points in t0031, t0032 and t0035, settling at 128 channels, but every one of those runs had the gate modulating a signal of element RMS 0.044 against a value stream of RMS 1.0, so the gate was deciding how much of an almost invisible contribution to admit. It now modulates a contribution at parity with v, which is a materially harder decision and plausibly one that wants more context. Widening to 256, the value that lost by 0.000668 under the old scale, tests whether that loss was conditional on the defect.

## t0092: discard

t0092 did not improve the prior champion: val_bpb=0.996374, delta=+0.027985. Hypothesis: The initialisation fix in t0087 improved the value-embedding path but simultaneously made its update quantisation 32 times coarser, and nothing has corrected for that. These tables are stored as bf16 parameters, cast in init_weights alongside wte. bf16 carries 8 mantissa bits, so its resolution scales with magnitude: near the old element scale of 0.044 the ulp was 0.00024, but at the corrected scale near 1.0 it is 0.0078. AdamW writes its update directly into that bf16 tensor, so any per-row update smaller than the ulp is silently rounded away. This bites hardest exactly where it matters: embedding gradients are sparse, so a given row is touched only when its token appears and its Adam update is often far below the nominal rate, and the warmdown drives the effective rate to a tenth of peak over the final half of the budget. Keeping the tables in fp32 while casting the gathered rows to the activation dtype preserves those small updates at no change to the forward computation.

## t0093: failed

t0093 ended as failed. Hypothesis: t0092 showed the value-embedding path is precision-limited but bought the fix in the most expensive place. Holding those tables in fp32 improved quality per step, loss 3.013 against the champion 3.029 at step 1000, yet throughput collapsed from 105ms to 211ms per step and the run lost 36 percent of its steps, finishing at 0.996374, worse than the original baseline. The cost came from tripling memory traffic on the largest parameter block, 50M parameters as fp32 weights plus fp32 moments. There is a cheaper place to spend that precision. _step_adamw allocates its moment buffers with torch.zeros_like(p), so a bf16 parameter silently receives bf16 moments, and the moments are where small updates must survive across steps: exp_avg holds a running average divided by the square root of exp_avg_sq, and both sit at an ulp of 0.0078 near the corrected embedding scale. Keeping parameters in bf16, so gathers and writes stay cheap, while allocating moments in fp32 should capture most of the quality effect at a fraction of the traffic.. Evidence: expected one nonce-bound locked evaluation, found 0; training process exited with code 1; training_seconds is missing from output

## t0094: discard

t0094 did not improve the prior champion: val_bpb=0.968509, delta=+0.000121. Hypothesis: Unchanged from t0093: the value-embedding path is precision-limited, as t0092 demonstrated by improving loss per step from 3.029 to 3.013 at step 1000, and the cheap place to spend that precision is the AdamW moment buffers rather than the parameters, because _step_adamw allocates them with torch.zeros_like(p) and a bf16 parameter therefore receives bf16 moments with an ulp of 0.0078 near the corrected embedding scale. t0093 did not test this. It aborted during compilation of adamw_step_fused with a fake-tensor failure on exp_avg.lerp_(grad, 1 - beta1_t): the moment buffer is now fp32 while grad remains bf16, and an in-place operation cannot promote its own destination. The diagnosis is a dtype contract inside the fused step, not a defect in the idea.

## t0095: discard

t0095 did not improve the prior champion: val_bpb=0.980817, delta=+0.012429. Hypothesis: The MLP nonlinearity is the last untouched component of the transformer block. Every other part has been measured: attention span and pattern, head width and count, value embeddings and their gate, the residual and skip scalars, depth, width and the MLP expansion ratio. The activation is squared ReLU, which is the modded-nanogpt lineage choice and is plausibly tuned, but it has never been tested here and it sits on the path carrying two thirds of the matrix FLOPs. GELU is the natural comparison: same parameter count, same FLOP count to within a rounding error, no change to any other quantity, and a materially different shape, being smooth and slightly negative near zero rather than exactly zero below the origin and quadratic above it. If squared ReLU is load-bearing rather than incidental this should regress and the block is then fully characterised.

## t0096: keep

t0096 improved the prior champion: val_bpb=0.967286, delta=-0.001103. Hypothesis: Weight averaging was dismissed early in this campaign on a premise that has since become false. The argument then was that the schedule anneals the rate to zero, so the final weights are already settled and Polyak averaging has nothing to denoise. That stopped being true at t0048, which introduced FINAL_LR_FRAC of 0.1 and was promoted: the model now finishes training at a tenth of peak rate, still taking meaningful steps, so its final parameters are a single noisy draw from a distribution rather than a converged point. Averaging the tail of the trajectory should land closer to the centre of that distribution than any individual endpoint. The cost is small and can be checked rather than assumed: the averaged copy is about 96M parameters, so each update moves roughly 1GB of memory traffic against a 105ms step, which is well under one percent.

## t0097: discard

t0097 did not improve the prior champion: val_bpb=0.967445, delta=+0.000159. Hypothesis: Weight averaging gained 0.001103, the largest single improvement in the last forty trials, and its window has not been tuned. The mechanism is specific: FINAL_LR_FRAC of 0.1 means the model never fully anneals, so its final parameters are one noisy draw rather than a converged point, and averaging the tail lands nearer the centre of that distribution. The averaging cost almost nothing in throughput, 2870 steps against the champion 2875, so the trade is nearly free and the only question is the window length. The decay sets that window: at 0.99 the effective horizon is about 100 steps, roughly the last 3 percent of a 2870-step run. If the endpoint noise is what is being removed, a longer horizon should remove more of it, up to the point where the average starts including weights from a materially higher learning rate. Raising the decay to 0.995 doubles the horizon to about 200 steps and tests that.

## t0098: discard

t0098 did not improve the prior champion: val_bpb=0.967527, delta=+0.000242. Hypothesis: The two elite lines are the same mechanism at two window lengths, and together they say the optimum lies at or below the shorter one. Weight averaging at a decay of 0.99, an effective horizon near 100 steps, gained 0.001103 over the unaveraged champion; widening it to 0.995 and roughly 200 steps gave back 0.000159. The reading that fits both is that what averaging removes is genuine endpoint noise from a model that never fully anneals, and that reaching further back mixes in weights from a materially higher learning rate which the schedule was deliberately moving away from. If that is right the optimum is short rather than long, and the untested direction is shorter still. A decay of 0.98 halves the horizon to about 50 steps and, with the 0.995 measurement, brackets the axis from both sides.

## t0099: discard

t0099 did not improve the prior champion: val_bpb=0.967290, delta=+0.000004. Hypothesis: Averaging changes what the schedule endpoint is for, and the endpoint has not been re-measured since averaging arrived. FINAL_LR_FRAC was raised from 0.0 to 0.1 in t0048, where it measured within noise and was promoted on a strict median; the reason a floor is normally costly is that the model stops settling and its final weights are left noisy. Weight averaging removes exactly that cost, and t0096 proved it does so here, gaining 0.001103 by averaging the tail of a run that never fully anneals. With the noise now absorbed by the average, a higher floor should be affordable and should buy real progress: the run spends its last 30 percent at a rate that decays toward the floor, so doubling the floor to 0.2 keeps the model learning materially faster through roughly 860 steps whose endpoint noise the EMA is already cleaning up. The two settings are complements rather than independent knobs.

## t0100: discard

t0100 did not improve the prior champion: val_bpb=0.969158, delta=+0.001872. Hypothesis: t0099 returned a null so exact it is a measurement in its own right: doubling the schedule floor from 0.1 to 0.2 changed val_bpb by four parts in a million, against a repeat noise of 1.3e-4. The reading is that weight averaging has decoupled final quality from where the schedule ends. Annealing normally serves two purposes, making progress at a useful rate and settling the model so its endpoint is not a noisy draw; the EMA introduced in t0096 now performs the second, and t0099 shows the first is insensitive to the floor across a factor of two. If the floor is genuinely not binding, the schedule is giving away rate for no return, and a much higher floor should let the model keep learning through the entire warmdown while the average absorbs the extra endpoint noise. Raising it to 0.4 tests where that stops being true.

## t0101: discard

t0101 did not improve the prior champion: val_bpb=0.968589, delta=+0.001303. Hypothesis: The warmdown fraction was measured before weight averaging existed, and averaging is exactly what made its cost avoidable. t0026 shortened the anneal from 0.5 to 0.35 and lost 0.000758, and t0020 lengthened it to 0.8 and lost 0.001467, so 0.5 was bracketed. But the reason a short anneal costs is that the model reaches the end of training less settled, and t0096 introduced an EMA that settles it, with t0099 then showing the schedule endpoint no longer influences the locked score across a doubling of the floor. If the endpoint is now handled by the average, the anneal is buying less than it did and the budget it consumes is better spent at full rate: shortening to 0.35 returns 15 percent of the run, roughly 430 steps, to the flat phase. This recipe has also shown repeatedly that it is expensive to deprive of full-rate steps, since a 2 percent warmup cost 0.001875 in t0061, which is the same quantity being bought back here.

## t0102: discard

t0102 did not improve the prior champion: val_bpb=0.968007, delta=+0.000721. Hypothesis: The anneal has been measured for length but never for shape. Its fraction of the budget is bracketed at 0.5, with 0.35 costing 0.000758 before averaging and 0.001303 after, and 0.8 costing 0.001467; its endpoint is bracketed with 0.2 an exact null and 0.4 costing 0.001872. Throughout, the decay itself has been linear in elapsed time, which is a choice rather than a necessity, and this recipe has proved unusually sensitive to how full-rate time is spent: a 2 percent warmup cost 0.001875 in t0061. A cosine anneal between the same two endpoints keeps the rate near its peak for longer at the start of the decay and flattens again as it approaches the floor, front-loading rate where the model still benefits from it and settling more gently at the end. Given that t0101 just showed the anneal does real work beyond endpoint settling, changing how that work is distributed is the natural remaining question.

## t0103: discard

t0103 did not improve the prior champion: val_bpb=0.974490, delta=+0.007204. Hypothesis: Averaging the endpoint of the trajectory paid 0.001103 in t0096, and the natural question is whether averaging the trajectory itself pays too. Lookahead does that: a slow copy of the weights follows the fast ones by a fraction alpha every k steps, and the fast weights are then reset onto the slow copy, so the optimizer explores from a point that is itself an average of recent iterates rather than from the latest one. It is a different intervention from the EMA, which only changes what is evaluated and never influences training, whereas this changes the path. The conditions here favour it: the run takes about 2870 steps at a batch of 2**17, which is the noisiest per-step gradient this campaign has used, and t0096 established that this model reaches its end as a noisy draw rather than a converged point. The cost is small and checkable, roughly 1GB of traffic every five steps against a 105ms step.

## t0104: discard

t0104 did not improve the prior champion: val_bpb=0.969680, delta=+0.002394. Hypothesis: t0103 tested Lookahead on every parameter and lost 0.007204, and its risk note named the likely cause in advance: Muon maintains a momentum buffer and a NorMuon per-row variance estimate keyed to the parameters it is updating, and Lookahead rewrites those parameters without touching either, so for five steps after each sync the orthogonalised update is correcting a trajectory that no longer exists. The pre-registered follow-up was a larger k, but that only makes the disruption rarer rather than testing the diagnosis. Restricting Lookahead to the AdamW groups tests it directly: those moments are plain per-coordinate exponential averages of gradients, which tolerate a parameter being moved under them far better than an orthogonalisation whose scale comes from a running variance estimate. The groups this covers are also the ones worth covering, since they hold wte, lm_head and the twelve value-embedding tables, roughly 58M parameters and the path this campaign has repeatedly found load-bearing.

## t0105: discard

t0105 did not improve the prior champion: val_bpb=0.968249, delta=+0.000963. Hypothesis: Logit magnitude is the most sensitive quantity this campaign has found, and only one instrument controls it. The output softcap is the single inherited default that proved suboptimal, and its curve is steep in both directions: 7 costs 0.005206, 10 was the optimum at the old batch, 13 is the optimum now, 30 costs 0.004637. Both failures converge on the same mechanism, damping the gradient into lm_head, and t0039 showed that a uniform learning-rate reduction cannot substitute for it because the cap acts selectively on tokens whose logits have grown large. A z-loss is the other selective instrument for the same quantity: it penalises the log-partition, so it pushes back on exactly the examples whose logits are collectively large, and it does so through the loss rather than by clipping the function. If softcapping this model is worth several thousandths of a bpb, a second, smoother control on the same quantity is the most promising remaining lever.

## t0106: failed

t0106 ended as failed. Hypothesis: The training context length has never been varied, and document masking has made most of it unused. Attention now costs about 62.9M of the roughly 314M per-token FLOPs, and 50.3M of that sits in the four full-context layers spanning 2048 tokens; the other eight layers see only 256. Since t0010 those full-context layers cannot read across a document boundary anyway, so the span they actually exploit is the length of the enclosing document rather than the row. Halving the training context to 1024 halves the cost of those four layers, removing about 25M FLOPs per token or 8 percent of the total, which should convert into roughly 6 to 8 percent more optimizer steps. Evaluation is untouched: prepare.evaluate_bpb builds its own loader at MAX_SEQ_LEN and the model handles a longer input at eval because the rotary table is precomputed for ten times the configured length.. Evidence: expected one nonce-bound locked evaluation, found 0; training process exited with code 1; training_seconds is missing from output

## t0107: discard

t0107 did not improve the prior champion: val_bpb=0.967646, delta=+0.000360. Hypothesis: Unchanged from t0106: with document masking confining attention inside documents, the four full-context layers exploit the enclosing document rather than the full row, so halving the training context to 1024 should free about 8 percent of per-token FLOPs and convert them into optimizer steps while evaluation continues at MAX_SEQ_LEN. t0106 did not test this. It aborted before training on the assertion that TOTAL_BATCH_SIZE divides tokens_per_fwdbwd, because that quantity is computed as DEVICE_BATCH_SIZE times MAX_SEQ_LEN rather than the training length; doubling the device batch to 128 therefore produced 262144 tokens per forward-backward against a total batch of 131072. I changed the model config and the dataloader but missed the third place the sequence length is used.

## t0108: discard

t0108 did not improve the prior champion: val_bpb=0.971588, delta=+0.004302. Hypothesis: The residual carry scalars are the last untested initialisation in the model. GPT.forward computes x as resid_lambdas[i] times x plus x0_lambdas[i] times x0 at every layer, so resid_lambdas sets how much of the accumulated stream each block inherits, and it is initialised to exactly 1.0, the identity. That value is correct by construction rather than by measurement, and it was chosen for a depth-8 model; the champion runs depth 12, so the stream now passes through half again as many blocks before reaching the head. This is the same class of quantity as the weight decay corrected in t0065 and the embedding skip examined in t0072: a per-layer setting whose cumulative effect changed when depth did. Damping the carry slightly to 0.95 compounds to about 0.54 across twelve layers, which is a material change to how much early-layer signal survives to the output.

## t0109: discard

t0109 did not improve the prior champion: val_bpb=0.967923, delta=+0.000637. Hypothesis: Decoupling the value embeddings changed what EMBEDDING_LR governs, and its value has not been re-measured since. Until t0089 that constant drove one AdamW group containing both transformer.wte and the twelve value-embedding tables, roughly 58M parameters, and 0.6 was confirmed optimal for that combination twice, at batch 2**18 in t0012 and again at 2**17 in t0053. t0089 then gave the value embeddings their own rate and measured their preference at 0.3, bracketed by t0090 at 0.15 and by the old shared 0.6. That is the inference: if the blended optimum over both groups was 0.6 while the larger of the two prefers 0.3, then wte alone must prefer something above 0.6, otherwise the blend could not have landed there. wte is now only 4.2M parameters and is normalised immediately in forward, so its rate is not constrained by the value stream any more. Raising it to 0.9 tests the inference.

## t0110: discard

t0110 did not improve the prior champion: val_bpb=1.019412, delta=+0.052126. Hypothesis: The output softcap is the only constant in this recipe whose optimum has been observed to move, and it has moved in one direction each time the model around it changed. It was bracketed at 10 at batch 2**18, then t0069 raised it to 13 and gained 0.000214 once lm_head became less sensitive at the smaller batch. Since then three further changes have altered what reaches lm_head: the value-embedding tables were rescaled by a factor of 22.6 in t0087, their learning rate was halved in t0089, and weight averaging in t0096 changed which weights are finally scored. Each of those makes the representation feeding the head better conditioned, which is the same condition that moved the optimum upward before. The cap is also the steepest axis measured here, so being 20 percent off it has cost thousandths of a bpb in both directions historically. Testing 16 checks whether the optimum has drifted again.

## t0112: discard

t0112 did not improve the prior champion: val_bpb=0.971295, delta=+0.004009. Hypothesis: This re-tests the softcap drift question that t0110 failed to answer. That node set the cap to 16 and returned 1.019412, worse than the untuned baseline, but the result is not attributable to the cap: it completed only 1342 optimizer steps against 2863 for the immediately preceding trial, and a foreign process occupying 104GB at full utilisation appeared on the pinned GPU partway through the run. A 53 percent step collapse is a contention signature, not a modelling effect, so the measurement was discarded and the campaign moved to an uncontended GPU. The underlying hypothesis stands: the output softcap is the only constant in this recipe whose optimum has been observed to move, rising from 10 to 13 in t0069 once lm_head became less sensitive, and three subsequent changes have further improved the representation feeding the head, namely the value-embedding rescale in t0087, its halved rate in t0089 and weight averaging in t0096. Testing 15 checks for further drift; 16 is not retested because that source already exists and would be rejected as a duplicate candidate.

## t0113: discard

t0113 did not improve the prior champion: val_bpb=0.975221, delta=+0.007935. Hypothesis: Both elite lines concern the endpoint of training, and one parameter of that machinery has never been measured. t0096 introduced weight averaging and gained 0.001103; t0099 then showed the schedule floor is inert across a doubling because the average now absorbs endpoint noise. The averaging window has been bracketed through its decay, 0.98 and 0.995 both worse than 0.99, but EMA_START has only ever been argued about, not tested. My argument was that it cannot bind: with a decay of 0.99 the effective horizon is roughly 100 steps while the averaging phase spans about 860, so the average forgets its starting point many times over. Moving the start from 0.7 to 0.5 nearly doubles the phase to about 1430 steps and tests that argument directly. If the reasoning is right this is an exact null, which is the outcome I expect.

## t0117: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0118: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0120: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0121: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0122: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0123: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0124: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0125: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0126: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0127: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0129: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0130: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0131: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0133: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0134: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0135: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0136: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0137: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0138: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0139: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0140: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0141: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0142: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0143: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0144: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0145: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0146: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0147: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0148: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0149: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0150: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0151: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0152: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0153: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0154: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0155: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0156: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0157: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0158: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0159: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0160: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0161: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0162: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0163: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0164: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0165: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0166: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0167: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0168: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0169: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0170: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0171: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0172: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0173: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0174: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0175: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0176: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0177: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0178: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0179: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0180: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0181: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0182: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0183: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0184: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0185: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0186: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0187: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0188: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0189: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0190: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0191: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0192: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0193: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0194: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0195: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0196: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0197: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0198: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0199: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0200: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0201: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0202: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0203: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0204: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0205: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0206: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0207: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0208: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0209: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0210: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0211: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0212: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0213: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0214: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0215: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0216: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0217: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0218: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0219: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0220: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0221: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0222: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0223: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0224: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0225: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0226: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0227: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0228: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0229: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0230: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0231: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0232: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0233: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0234: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0235: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0236: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0237: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0238: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0239: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0240: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0241: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0242: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0243: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0244: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0245: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0246: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0247: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0248: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0249: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0250: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0251: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0252: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0253: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0254: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0255: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0256: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0257: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0258: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0259: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0260: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0261: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0262: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0263: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0264: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0265: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0266: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0267: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0268: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0269: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0270: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0271: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0272: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0273: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0274: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0275: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0276: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0277: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0278: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0279: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0280: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0281: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0282: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0283: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0284: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0285: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0286: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0287: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0288: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0289: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0290: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0291: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0292: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0293: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0294: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0295: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0296: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0297: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0298: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0299: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0300: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0301: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0302: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0303: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0304: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0305: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0306: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0307: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0308: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0309: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0310: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0311: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0312: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0313: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0314: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0315: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0316: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0317: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0320: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0321: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0322: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0323: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0324: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0325: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0326: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0327: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0328: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0329: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0330: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0332: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0333: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0334: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0335: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0336: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0337: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0338: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0339: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0340: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0341: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0342: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0343: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0344: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0345: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

## t0346: agent_error

agent_error: agent exited with code 1; agent did not produce its structured result; agent result is missing non-empty hypothesis; agent result is missing non-empty change_summary; agent result is missing non-empty expected_val_bpb_effect; agent result is missing non-empty risk

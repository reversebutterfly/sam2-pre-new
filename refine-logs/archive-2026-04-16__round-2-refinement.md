# Round 2 Refinement

## Changes: Add memory-reset controls for clean and insert-only

### Added rows:
- **A0r**: clean + memory reset at f4 (normalize reset effect)
- **A5r**: insert-2-adv + memory reset at f4 (key causal test for insertion)

### Moved to appendix: A6, A6b (density trend, not core proof)

## Final Table A: Mechanism Decomposition (CORE)

Attack window: f0-f3. Evaluate ONLY f4-f14.

| ID | Condition | Attack (f0-f3) | ε | Memory reset at f4? | Core purpose |
|---|---|---|---|---|---|
| A0 | clean | — | 0 | No | upper bound |
| A0r | clean+reset | — | 0 | **Yes** | normalize reset effect |
| A1 | frame0-only | perturb f0 | 2/255 | No | conditioning hijack |
| A2 | orig-2 | perturb f1,f2 | 4/255 | No | memory poisoning |
| A3 | orig-2-strong | perturb f1,f2 | 8/255 | No | budget-matched with A5 |
| A4 | frame0+orig-2 | perturb f0+f1,f2 | 2+4/255 | No | conditioning+memory |
| A5 | insert-2-adv | insert 2 adv after f1,f3 | 8/255 | No | adversarial insertion |
| A5b | insert-2-benign | insert 2 clean interp after f1,f3 | 0 | No | benign control |
| A5r | insert-2-adv+reset | insert 2 adv after f1,f3 | 8/255 | **Yes** | causal: insertion → memory? |
| A7 | hybrid | perturb f0,f1,f2 + insert 2 | mixed | No | full method |
| A8 | hybrid+reset | perturb f0,f1,f2 + insert 2 | mixed | **Yes** | full method causal |

### Decisive Evidence Pattern

```
IF:  A5 >> A5b (adversarial > benign insertion)
AND: A5r ≈ A0r (memory reset kills insertion effect)
AND: A5 >> A0 (insertion alone causes degradation)

THEN: Adversarial frame insertion corrupts SAM2 via memory poisoning. QED.
```

Additional insights:
- A1 vs A0: how powerful is conditioning-frame attack alone?
- A3 vs A5: perturbation vs insertion at matched budget
- A7 vs A4 + A5: is hybrid synergistic?
- A8 vs A7: full method is memory-mediated

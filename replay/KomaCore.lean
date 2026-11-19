import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace Koma

/-- Odd-sublattice particle-hole complements both occupations. -/
def centered (u v : ℚ) := (u - 1/2) * (v - 1/2)

theorem centered_ph (u v : ℚ) : centered (1-u) (1-v) = centered u v := by
  unfold centered
  ring

theorem bare_split (u v : ℚ) : u*v = centered u v - (1-u-v)/2 + 1/4 := by
  unfold centered
  ring

theorem bare_odd_vacuum_mismatch (lam : ℚ) (h : lam ≠ 0) :
    lam * (0*0) ≠ lam * ((1-0)*(1-0)) := by
  simpa using Ne.symm h

/-- The off-diagonal entries required by the two reflected Majoranas. -/
theorem cross_symmetric_iff (t : ℚ) : t/2 = -t/2 ↔ t = 0 := by
  constructor
  · intro h; linarith
  · intro h; rw [h]; norm_num

/-- Final margin in `src/proofs.tex`, Section 5. `r` and `delta` are the analytic IR error terms. -/
theorem uniform_margin (e r delta m : ℝ)
    (he : 499/1000 ≤ e) (hr : r^2 ≤ (49/100)*(e + 6/10000))
    (hd : delta ≤ 2/10000) (hm : e - delta - r ≤ m) : 3/1000 ≤ m := by
  have h₁ : 0 ≤ (e - 499/1000)^2 := sq_nonneg _
  have h₂ : (49/100)*(e + 6/10000) < (e - 32/10000)^2 := by
    nlinarith only [h₁, he]
  by_contra! hm'
  have h₃ : 0 < r - (e - 32/10000) := by linarith
  have h₄ : 0 ≤ r + (e - 32/10000) := by linarith
  have h₅ := mul_nonneg (le_of_lt h₃) h₄
  nlinarith only [h₂, hr, h₅]

#print axioms centered_ph
#print axioms bare_split
#print axioms bare_odd_vacuum_mismatch
#print axioms cross_symmetric_iff
#print axioms uniform_margin

end Koma

# How to Run Task 2 — Step by Step (no MATLAB experience assumed)

## 0. Before you start

You need **MATLAB** with the **Fuzzy Logic Toolbox** installed (Global
Optimization Toolbox is *not* needed — the GA/PSO here are hand-written).

- If your university gives you MATLAB (most do, via Coventry/Softwarica),
  either open the **desktop MATLAB app**, or go to
  **matlab.mathworks.com** and sign in with your university email to use
  **MATLAB Online** in a browser — both work the same way for this.
- Check you have the Fuzzy Logic Toolbox: in MATLAB, type `ver` and press
  Enter. Look for "Fuzzy Logic Toolbox" in the list. If it's missing, use
  the **Add-Ons** button (top toolbar, puzzle-piece icon) to install it —
  it's free with your university license.

Every step below happens inside MATLAB's **Command Window** (the panel
where you type commands and press Enter) unless said otherwise.

---

## PART 1 — Build and inspect the Fuzzy Controller

**Step 1: Open the folder in MATLAB.**
In the Command Window, navigate into `part1_flc`:
```matlab
cd part1_flc
```
Alternative: in the **Current Folder** panel (on the left in MATLAB), navigate to
`part1_flc` and double-click into it.

**Step 2: Build the controller.**
Type:
```matlab
build_flc
```
Press Enter. You should see text printed like "Saved thermostat_flc.fis"
and a checklist of two things to screenshot. Nothing visual pops up yet —
that's expected, this step just builds and saves the controller.

**Step 3: Take the two required screenshots.**
Copy-paste each line below into the Command Window, press Enter, wait for
a window to open, then take a screenshot of that window (on Mac:
`Cmd+Shift+4` then click-drag over the window; on Windows: `Win+Shift+S`).

```matlab
fuzzyLogicDesigner(fis)
```
A window opens showing your two inputs, one output, and the rule list.
Screenshot it. Save the image somewhere you'll remember, e.g.
`task2_flc/part1_flc/screenshot_fis_editor.png`. Close the window when done.

```matlab
ruleview(fis)
```
A window opens with sliders for the two inputs. Drag the top slider
(TempError) to about **-8** and the second slider (RateChange) to about
**-1.5** — you'll see the rule rows highlight and the output on the
right update. Screenshot it (name it e.g. `screenshot_ruleview_cold.png`).
Then try TempError ≈ **7**, RateChange ≈ **1** for a second, different
screenshot (`screenshot_ruleview_hot.png`) showing different rules firing.
Close the window when done.

**Step 4: Generate the automatic plots.**
Back in the Command Window (make sure you're still in the `part1_flc`
folder), type:
```matlab
analyze_flc
```
Press Enter. This one runs on its own — no clicking needed. When it
finishes, look in the `part1_flc` folder (in the **Current Folder** panel
on the left, or in Finder/File Explorer). You'll now see, all generated
automatically:

| File | What it is / which mark it serves |
|---|---|
| `membership_functions.png` | MF plots for both inputs + output (design evidence) |
| `control_surface.png` | the required control-surface plot |
| `scenario_results.csv` | controller output at 6 representative operating points |
| `rule_activation.csv` / `.png` | **firing strength of each of the 15 rules** at two contrasting operating points -- the numeric version of the rule-viewer screenshot |
| `closed_loop_response.png` | **closed-loop step response** of the room under the FLC vs a classical linear PD controller |
| `closed_loop_metrics.csv` | settling time, overshoot, steady-state error, IAE, energy for both controllers |

The last three are what answer the brief's *"demonstrating how the
controller achieves the specified behaviours in relation to an operational
scenario"* (7 marks): the static probe points show what the controller
commands at an instant, the closed-loop run shows what it actually does
over time. Those files plus your two screenshots from Step 3 are
everything Part 1 needs.

Note the console also prints the closed-loop comparison table -- copy that
output into the report, it is not otherwise saved as an image.

---

## PART 2 — Optimise the Controller with a Genetic Algorithm

**Step 1: Move to the Part 2 folder.**
```matlab
cd ../part2_ga_tuning
```

**Step 2: Generate the training dataset.**
```matlab
generate_reference_dataset
```
Press Enter. You'll see a short message printed confirming a file called
`reference_dataset.csv` was saved. This is a made-up (but realistic)
dataset the GA will use to judge whether its tuning is any good.

**Step 3: Run the Genetic Algorithm.**
```matlab
ga_tune_flc
```
Press Enter and **wait** — this runs 60 generations of a GA and will take
somewhere between a few seconds and a couple of minutes depending on your
computer. You'll see two numbers printed: the untuned FLC's error, and the
GA-tuned FLC's error (lower is better) — check the tuned one is lower than
the untuned one. Two chart windows will also pop up automatically
(convergence curve, and before/after membership function shapes) and get
saved as `ga_convergence.png` and `before_after_mfs.png` in that folder.
No screenshots needed here — everything saves itself.

**Step 4: Check what came out.**
In the `part2_ga_tuning` folder you should now see:

| File | What it is |
|---|---|
| `reference_dataset.csv` | the 300 (input, target-output) examples the GA is scored against |
| `thermostat_flc_tuned.fis` | the tuned controller |
| `ga_convergence.png` | GA convergence curve (best MSE per generation) |
| `before_after_mfs.png` | membership functions before vs after tuning |
| `before_after_surface.png` | control surface before vs after tuning |
| `closed_loop_before_after.png` / `.csv` | closed-loop response and metrics, untuned vs tuned |
| `ga_tuning_summary.csv` | the two headline MSE numbers |
| `best_chromosome.csv` | the winning 6 genes |

The closed-loop before/after is worth emphasising in the report: the GA's
fitness is *open-loop* MSE against the reference dataset, so the
closed-loop result is an evaluation the GA never optimised for. An
improvement there is evidence the tuning genuinely improved the
controller, not just that the GA fitted its own objective.

**Rough numbers to sanity-check against.** I re-implemented this whole
chain in Python to test it before you run it, and got untuned MSE ~214,
GA-tuned ~134 (a ~37% reduction). MATLAB's random number generator and
centroid discretisation differ from mine, so your exact figures will not
match -- but if you see a reduction of roughly that order, it is working.
Report *your* MATLAB numbers, not these.

---

## PART 3 — Compare Two Optimisation Algorithms

**Step 1: Move to the Part 3 folder.**
```matlab
cd ../part3_optimizer_comparison
```

**Step 2: Run the comparison.**
```matlab
run_comparison
```
Press Enter and **wait longer this time** — this runs both algorithms 15
times each, on 2 functions, at 2 problem sizes (so 4 combinations x 2
algorithms x 15 runs = 120 total optimisation runs). This could take
several minutes to tens of minutes depending on your computer — you'll
see progress lines print like "Done: sphere, D=2" after each combination
finishes, so you know it's still working, not stuck.

**Step 3: Check what came out.**
When it finishes you'll see a results summary printed in the Command
Window (mean/std/best/worst for GA and PSO on each function/dimension). In
the folder you'll also find `optimizer_comparison_results.csv` (open this
in Excel or Numbers — it's the exact table to paste into your report) and
these files:

| File | What it is |
|---|---|
| `optimizer_comparison_results.csv` | the summary table to paste into the report: mean/std/best/worst/median **error to the known optimum**, plus a Wilcoxon rank-sum p-value comparing GA against PSO |
| `optimizer_per_run_errors.csv` | all 15 individual run results, for the appendix |
| `convergence_<func>_D<n>.png` | 4 convergence plots (log scale, median of 15 runs), both algorithms on each chart |
| `boxplot_<func>_D<n>.png` | 4 box plots showing the spread of the 15 final errors |

Results are reported as **error to the optimum**, `f(x) - f(x*)`, which is
the CEC'2005 convention and is what makes your numbers comparable with
published results. Smaller is better, and 0 is a perfect solve.

**Rough numbers to sanity-check against.** From my Python port of the same
GA and PSO: PSO wins clearly on Sphere (error ~1e-12 at D=2, ~6e-4 at
D=10, versus GA's ~5e-6 and ~2.5); GA wins clearly on Rastrigin at D=10
(~8.7 versus PSO's ~19.3). That contrast -- PSO converges faster on the
smooth unimodal function, GA resists the many local optima of the
multimodal one better at higher dimension -- is the interesting finding to
build the Part 3 discussion around. Again: report your own MATLAB numbers.

**If `ranksum` is missing**, the p-value column will read `NaN` -- that
function needs the Statistics and Machine Learning Toolbox. Everything
else still works; just say in the report that the comparison is
descriptive, or install that toolbox via Add-Ons.

---

## If something goes wrong

- **"Undefined function 'mamfis'" or similar** → the Fuzzy Logic Toolbox
  isn't installed. Go to Add-Ons (puzzle icon) and install it.
- **"File not found" errors** → you're not in the right folder. Use `cd`
  (Step 1 of whichever part you're on) again, or check the Current Folder
  panel shows you're inside the correct `part1_flc` / `part2_ga_tuning` /
  `part3_optimizer_comparison` folder.
- **Anything else that errors** → copy the exact red error text and send it
  to me — I can't run MATLAB myself to test this in advance, so this is
  the fastest way for me to help you fix it.

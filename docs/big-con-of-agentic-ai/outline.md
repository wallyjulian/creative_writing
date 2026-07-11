# The Big Con of Agentic AI — Outline

## Audience and rhetoric
- Format: Quarto reveal.js (HTML slides), R for all charts.
- Audience: Working deck (for personal reference / future-self). Rigor > polish, more
  text tolerated than a stage performance, preserves nuance rather than flattening it.
  Rhetorical balance: ~60% logos, 30% ethos, 10% pathos.
- Aesthetic: clean, minimal, professional. Single restrained accent color, no
  decorative personality, generous white space. Custom (not default-reveal.js)
  but deliberately quiet.
- Closing line: "Do the thinking first."

## Theme
- Palette: off-white background, charcoal text, one muted slate-blue accent for
  rules/titles/emphasis, a muted red reserved only for the one or two numbers
  that should visually jump out (the token-waste stat).
- Frame-title style: thin left rule in the accent color, bold dark title, no
  background tint, no icons.
- Custom SCSS theme file, not Quarto's default reveal look.

## The arc

### Act I — Setup (slides 1–4)
1. Title slide: "The Big Con of Agentic AI" / "What over-dependence on
   consultants teaches us about delegating cognition to machines"
2. Opening hook (stat-led): "For every $1 spent on AI tokens, 18 cents
   creates user-facing value. 44 cents pays to fix bugs the AI itself
   introduced." → Fig 1 (bar chart)
3. Stakes: agentic AI is repeating a known structural trap, not a new one
4. Roadmap: consulting's con → AI's version → three levels of cost → a third path

### Act II — Investigation (slides 5–15)
5. The original con: consulting extracts value from dependency, not fraud
   (Mazzucato & Collington, *The Big Con*)
6. The mechanism: "unlearning by not doing" — the less you row, the less you
   steer (simple two-box/arrow diagram, HTML, not a data chart)
7. Comparative advantage is rational until it eats identity-defining work
8. Table 1: the consulting-to-AI structural parallel, mechanism by mechanism
9. Subsidized pricing today is designed to become lock-in tomorrow
10. Real-world costs are already landing (Uber's 4-month AI budget, Microsoft
    cutting internal licenses, Salesforce's ~$300M Anthropic spend) — small
    comparison table, not a chart (units are incompatible across the three)
11. Vendors are reframing AI from augmentation to labor replacement (larger
    addressable market logic + the advisor conflict of interest)
12. Individual cost: cognitive debt replaces independent thinking
13. Organizational cost: the "oversight tax" — paying for both the system and
    a reviewer whose independent judgment is quietly eroding
14. Societal cost: HITL as accountability sink + Fig 2 (global compute
    concentration: US + China ≈ 90%)
15. Devil's advocate: banning AI and uncritical adoption are both dead ends

### Act III — Resolution (slides 16–17)
16. The third path: at every task, ask whether AI is building capacity or
    substituting for it — one framework table across individual/org/societal
17. Closing, full-bleed: "Do the thinking first."

## Figures and tables (R, self-contained chunks)
- Fig 1: token-spend breakdown bar chart (18% value / 44% bug-fixing / 38% other)
- Table 1: consulting-vs-AI structural parallel (kable, booktabs-style)
- Table 2: real-world cost anecdotes (Uber / Microsoft / Salesforce)
- Fig 2: global compute concentration bar (US+China ~90% vs rest of world)
- Table 3: third-path framework (level → risk → deliberate countermeasure)

~17 slides total.

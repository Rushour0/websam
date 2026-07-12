<!--
  DESIGN.md — this project's pinned design taste bar.
  Detected by the ux-design-engineer persona's Frame phase from the real, already-shipped
  components (Timeline.tsx, Toolbar.tsx, MediaLibrary.tsx, src/components/ui/*, tailwind.config.ts,
  src/index.css) and committed at apps/studio's root. Source of truth for design consistency and
  the checklist every browser-verify pass runs against.
-->

# DESIGN.md — Design Taste Bar (apps/studio)

## 1. Detected design system

- [x] **Existing system detected** — source: `apps/studio/tailwind.config.ts` (shadcn CSS-var color
      mapping, `--radius`-derived border radii) + `apps/studio/src/index.css` (`:root`/`.dark` HSL
      custom properties) + `src/components/ui/{button,slider,toggle-group}.tsx` (shadcn primitives,
      `class-variance-authority` variants) + lived usage in `Timeline.tsx`/`Toolbar.tsx`/
      `MediaLibrary.tsx`. Follow it; do not invent parallel tokens or a new accent color.

Notes: React 18 + Tailwind 3 + shadcn/ui (CSS-vars mode) + `lucide-react` icons +
`class-variance-authority` for variants + `cn()` (clsx+tailwind-merge) helper in `src/lib/utils.ts`.
**The palette is the UNTOUCHED shadcn "neutral/zinc" default** — `--primary` is near-black
(`240 6% 10%`), and `--secondary`/`--muted`/`--accent` all resolve to the SAME light-gray
(`240 5% 96%`) with no distinct hue anywhere. Normally an untouched shadcn default is a slop tell
(craft-bar.md), but here it's the project's own deliberate, already-consistent system across every
existing panel — the fix is hierarchy/spacing/grouping discipline within that system, not introducing
a new brand accent color unprompted. Only `--destructive` (red, `0 84% 60%`) breaks monochrome, used
exclusively for delete/error — that is the project's one semantic (not decorative) color signal.

## 2. Color roles

| Role        | Token name              | Value (light)      | Usage note                                        |
| ----------- | ------------------------ | ------------------- | -------------------------------------------------- |
| background  | `--background`           | `0 0% 100%`          | app canvas / lowest layer                          |
| surface     | `--card`                 | `0 0% 100%`          | `PropertiesPanel`/panels (`bg-card`)                |
| primary     | `--primary`               | `240 6% 10%` (near-black) | primary actions, active/selected state           |
| secondary   | `--secondary`             | `240 5% 96%`          | secondary buttons, low-emphasis fills              |
| muted       | `--muted` / `--muted-foreground` | `240 5% 96%` / `240 4% 46%` | section labels, helper/caption text          |
| border      | `--border`                | `240 6% 90%`          | dividers between sections, input outlines          |
| destructive | `--destructive`           | `0 84% 60%`           | delete/irreversible actions ONLY — the one accent  |
| accent      | `--accent`                 | `240 5% 96%`          | hover fill on ghost buttons/rows (`hover:bg-accent`) |

No separate brand/primary-hue accent exists or should be introduced — `--destructive` red is the
only color that isn't grayscale, and it is reserved for delete/error, never decoration.

## 3. Type scale (as actually used across Toolbar/Timeline/MediaLibrary/PropertiesPanel)

| Step    | Size        | Weight             | Usage                                             |
| ------- | ----------- | ------------------- | -------------------------------------------------- |
| section-label | `text-xs` (12px), `uppercase tracking-wide` | `font-semibold` | Section headers ("Clip", "Objects", "Export") — `text-muted-foreground` |
| body    | `text-sm` (14px)   | `font-medium` / regular | Primary row text (filenames, object labels)   |
| caption | `text-[11px]`/`text-xs` | regular         | Timeline ruler ticks, helper text, durations       |
| numeric input | `text-sm`     | regular             | Trim in/out steppers                                |

Font family: system default (no custom `--font-sans` declared) — Tailwind's default stack.

## 4. Spacing rhythm

- Base unit: **4px** (Tailwind default scale, used directly — `p-2`, `p-3`, `gap-1`, `gap-2`).
- Steps actually in use: `1 (4px) · 1.5 (6px) · 2 (8px) · 3 (12px)` — sections use `p-3` +
  `space-y-2`; rows use `gap-2`; icon-button clusters use `gap-1`.
- Rule: snap to these steps; a new section follows the existing `space-y-2 border-b border-border p-3`
  pattern already used by every `PropertiesPanel` section — do not introduce a new container padding.

## 5. Radii

| Step | Value                    | Usage                                    |
| ---- | ------------------------- | ------------------------------------------ |
| sm   | `calc(var(--radius) - 4px)` = 4px | small chips                        |
| md   | `calc(var(--radius) - 2px)` = 6px | buttons, inputs — the default (`rounded-md`) |
| lg   | `var(--radius)` = 8px      | not currently used in studio panels        |
| full | `9999px`                   | color swatches (`rounded-full`), slider thumb |

## 6. Motion budget

- Default duration: Tailwind default `transition-colors` (~150ms, no explicit duration override
  anywhere in the codebase).
- Default easing: browser default (no custom `ease-*`/`cubic-bezier` found).
- Max duration budget: 200ms — nothing in this dense, keyboard-first editor should feel slow.
- [x] Respect `prefers-reduced-motion`: only `transition-colors`/opacity today (no transform/layout
      animation), which is already reduced-motion-safe; do not add anything heavier without a
      `motion-reduce:` guard.
- Note: motion here is limited to color/opacity state transitions (hover, disabled, focus ring) —
  no entrance animation, no decorative loops anywhere in the app. Keep new controls (playback bar,
  volume) consistent: state changes only, never decorative.

## 7. Chosen intentional aesthetic

Dense, high-contrast, near-monochrome shadcn-neutral with a single reserved destructive-red accent;
keyboard-and-mouse dual-input editor chrome (closest anchor: Linear's restraint + Stripe's density).

## 8. Do / Don't (anti-slop)

**DO**

- [x] Match the tokens above — no ad-hoc colors, sizes, or radii; no new accent hue.
- [x] Establish hierarchy in greyscale first (it already mostly is — the palette has no hue to lean on).
- [x] Use `lucide-react` icons only, consistent `h-3.5 w-3.5` / `h-4 w-4` sizing per existing usage.
- [x] Make motion purposeful — color/opacity only, matching what's already there.
- [x] Group related actions visually (this is PropertiesPanel's actual gap — actions currently float
      inline with no grouping/rhythm distinct from content).

**DON'T**

- [ ] No indigo/violet gradients or the default "AI startup" palette — n/a, already avoided.
- [ ] No templated 3-equal-card feature grid — n/a, this is app chrome not a marketing page.
- [ ] No emoji as UI icons — already lucide-only, keep it that way.
- [ ] No gradient text / rainbow headings.
- [ ] Do NOT "fix" the untouched-shadcn-palette smell by inventing a new brand color — that would
      break consistency with Timeline/Toolbar/MediaLibrary, which is the worse failure here.

## 9. Accessibility floor

- [ ] Contrast: text ≥ WCAG AA 4.5:1; large text & UI components ≥ 3:1 — verify
      `text-muted-foreground` (`240 4% 46%` on white) meets 4.5:1 for body-sized use, not just captions.
- [ ] Visible focus state on every interactive element — `Button`'s `focus-visible:ring-1
      focus-visible:ring-ring` already covers standard buttons; any new custom control (scrub bar,
      volume slider) must carry the same focus-visible ring treatment.
- [ ] Fully keyboard-navigable; logical tab order; no keyboard traps — new PreviewCanvas playback
      controls must be real focusable elements (`<button>`/`<input type="range">`), not
      click-only `<div>`s.
- [ ] Semantic HTML + ARIA — `role="slider"`/`aria-label`/`aria-valuenow` on any hand-rolled range
      control (Timeline.tsx's `TrimHandle` already sets this precedent, follow it).
- [ ] Respects `prefers-reduced-motion` (see §6).
- [ ] Hit targets ≥ 24×24px minimum (existing icon buttons are `h-7 w-7`/`h-9 w-9` — stay in that range).

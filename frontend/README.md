# Client — Brain MRI Diagnostic Assistance

Next.js 16 frontend for the Python pipeline in the parent directory. It talks to the
FastAPI backend over REST; it contains no model code and no image processing of its own.

## Running

The backend must be running first — the client is a view onto it and has nothing to show
without it.

```bash
# from the project root, in one terminal
./run.sh serve      # FastAPI on http://localhost:3000
```

```bash
# in another
./run.sh client     # this app on http://localhost:5713
```

Or directly: `npm install && npm run dev`.

## Configuration

`.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:3000
```

The `NEXT_PUBLIC_` prefix is required: the fetch calls run in the **browser**, not on the
Next.js server. That is deliberate — proxying uploads through Node would copy every scan
into a second process and buffer the multi-megabyte base64 responses twice. Going direct
keeps the Python backend the only component that ever holds patient data.

If the variable is absent the client falls back to `http://localhost:3000`, so a fresh
clone works without any setup.

## Pages

| route | what it does |
|---|---|
| `/` | upload a slice, switch between Detection and Typing, see the verdict, per-class probabilities, all six pipeline stages and every measurement |
| `/performance` | held-out test metrics for either model: confusion matrix, per-class recall, and what the CNN branch contributed |

## Structure

```
src/
├── app/
│   ├── layout.tsx          shell, nav, the disclaimer that appears on every page
│   ├── globals.css         design tokens
│   ├── page.tsx            the analysis page
│   └── performance/        the metrics page
├── components/
│   ├── UploadZone.tsx      drag-and-drop, with client-side validation
│   ├── VerdictCard.tsx     headline result and which branch decided it
│   ├── ProbabilityBars.tsx full per-class distribution, CNN opinion marked separately
│   ├── StageGallery.tsx    the six pipeline stages, click to enlarge
│   ├── MeasurementPanel.tsx region geometry, anatomical location, enhancement gain
│   ├── ConfusionMatrix.tsx binary and 4-class variants
│   ├── TaskToggle.tsx      Detection / Typing switch
│   ├── ErrorPanel.tsx      shows the backend's suggested fix, not just the failure
│   └── HealthBadge.tsx     live backend status in the header
└── lib/
    ├── api.ts              typed fetch client with structured errors
    └── types.ts            response shapes mirroring backend/src/braintumor/api.py
```

## Notes on the design

**Dark by default, and not for fashion.** MRI slices are grayscale and Stage 3 of the
pipeline exists to recover low-contrast detail. A bright interface around a dark scan raises
the viewer's adaptation level and flattens exactly that detail. Radiology reading rooms are
dimmed for the same reason. Colour is reserved for meaning — a verdict, a class bar, an
error — so a coloured pixel here always denotes something.

**The stage gallery is the argument, not decoration.** The claim that a hybrid CV+ML system
is auditable in a way a raw CNN is not only holds if the intermediate stages are visible. A
clinician can see *which* region was measured and reject the call when the segmenter has
latched onto an artefact.

**The whole probability vector is shown, never just the winner.** On the typing task a
52/41 split between two types is a case for a human to look at closely, and it is
indistinguishable from a 95/2 case if the interface only prints the argmax.

# Atlas

An interactive 3D part explorer for the hardware in this repo and for the two
head-mounted displays the epilepsy accessibility project targets. Modelled on
[ashemag/human-atlas](https://github.com/ashemag/human-atlas): orbit the model,
click any part, toggle subsystem layers, scrub an exploded view, search, and read
what each part is and what it does.

```sh
npm ci
npm run dev        # http://localhost:3017
```

## The three devices

| Tab | What it is | Geometry |
|---|---|---|
| **FollowCam** | The Pod v4 pan pod — 26 parts, 12 of them printed | **Exact.** Exported from `cad/blender/pod_v4/build_v4.py`, the same script that writes the printable STLs |
| **Quest 3** | The hackathon display — 28 parts | Block schematic |
| **X3 Pro** | The roadmap AR glasses — 19 parts | Block schematic |
| **Quest 3 vs X3 Pro** | Which capability survives the move to glasses | — |

### On the two kinds of geometry

The FollowCam model is dimensionally exact because it comes out of the CAD that
produces the parts. Nothing equivalent exists for a Quest 3 or an X3 Pro — there is
no public CAD, and a teardown photograph is not a mesh. So those two are **block
schematics**: each component is a primitive sized and placed to be roughly right and
honest about what sits where and what it connects to. Envelopes, component counts and
published specifications are real and sourced on the part; internal placement is
representative. The app labels them `block schematic` and refuses to draw dimension
lines on them, because a leader line pointing at an invented edge would be a lie.

## Regenerating geometry

```sh
blender --background --python scripts/export_atlas_glb.py   # FollowCam, from build_v4.py
blender --background --python scripts/build_glasses.py      # Quest 3 + X3 Pro
```

Each writes a GLB into `public/models/` and a manifest into `src/generated/`. The
manifest carries system, explode vector, bounding box and triangle count; the prose
lives in `src/data/`. Nothing is duplicated between the two, so the atlas cannot drift
from the model — change a dimension in `build_v4.py` and it changes here.

Two things the exporters handle that are easy to get wrong:

- **Node names.** Three's `GLTFLoader` strips `. : / [ ]` from node names, so Blender's
  `leg.001` and `1/4-20` arrive as `leg001` and `1420` and silently fail to match.
  Both scripts rename objects to a safe `node` before export and assert uniqueness.
- **Coordinate space.** The GLB is exported Y-up while Blender authors Z-up, so every
  manifest carries both `explode` (Blender mm, matching the README tables) and
  `explodeGl` (glTF metres, what the viewer uses).

## Photosensitivity

Both headsets ship a photosensitive-seizure warning, and on retail hardware it is not
removable. The atlas does not try to get around it — it maps the display stack so the
actual engineering question can be answered: which components put flicker into the eye,
and which ones the app controls.

Turn on **Colour by flicker role** on either device, or open the **Flicker chain** view:

- **red — source.** Emits or modulates light and can flicker. The Quest's LED backlight
  is strobed at the refresh rate; the X3 Pro's microLED engines dim by PWM.
- **teal — lever.** Where you can intervene. The XR2's refresh rate is requested by the
  app and the display accepts any integer from 72 to 207 Hz; the LCD is where shader-side
  luminance and flash-area clamping lands.
- **blue-grey — path.** Light passes through it. The pancake stack caps peak brightness
  at about 100 nits, which is protective; the X3 Pro's waveguide is additive and cannot
  darken anything, which is the whole reason the Quest is the hackathon device.

Per-part notes carry the reasoning and cite where contested numbers came from.

## Stack

Vite, React 19, Three.js, hand-written CSS. No UI framework — the whole dependency tree
is six packages, which matters when the thing has to build the night before a demo.

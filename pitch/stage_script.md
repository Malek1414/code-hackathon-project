# FollowCam — stage script (~2:30)

Setup: rig assembled on stage, laptop linked, ball ready. 30s combined
video queued muted. Malek speaks, Sammy drives demo + takes the analytics
section.

---

**[0:00 — cold open, DEMO FIRST]** *(Sammy rolls the ball / walks it
across the stage; the camera visibly follows)*

Don't look at me — look at the tripod. That's a twenty-euro robot
cameraman following the ball, live, right now.

**[0:20 — the problem]**

Ninety percent of amateur games are never filmed. Not because nobody
wants them — because an auto-tracking camera costs two thousand euros
plus a subscription. So kids' games, regional leagues, women's leagues:
no footage, no stats, no highlights.

**[0:40 — the product]**

FollowCam is a retrofit. Two 3D-printed parts — a ring that clamps the
tripod you already own, and a fork arm on a metal-gear servo driven by
an Arduino. Your phone sits in the head; our app tracks the target
on-device and steers the servo. A hundred-degree sweep covers a full
court from the scorer's table. Total hardware: about twenty euros.

**[1:05 — the market]**

Veo and Pixellot: two thousand plus, locked hardware, priced for clubs
that already film. Pivo: consumer gadget that follows one person — and
the data dies in the video. We're the third category: cheapest possible
capture, and the video is just the input.

**[1:25 — the data layer]** *(Sammy: dashboard on screen)*

Because the same camera keeps the stats. Today — today — our pipeline
took real gym footage, tracked every player, read jersey numbers, mapped
them onto a 2D court, and computed the score you see on this dashboard:
shots, points, possession timeline, streaming as a live overlay. Points,
rebounds, assists is the ladder we're climbing — and heart-rate sync
tells a player when their pulse left the zone and their mistakes spiked.

**[1:55 — how the day actually went]**

Honest build story: our printed servo pocket was sized for the small
servo, and the good one didn't fit — the design's zip-tie ledges saved
us in five minutes. One jumper wire was dead. The camera preview shipped
black once. Every fix is a commit — the whole day is public on GitHub as
of an hour ago.

**[2:15 — close]**

One servo, two printed parts, the phone in your pocket — and every game
finally counts. FollowCam: film the game, keep the stats, twenty euros.
Thank you.

---

## Q&A ammo

- **"Does it work outdoors / other sports?"** Tracking is class-based —
  swap the target class. The mount fits any pan-handle tripod. Handball
  footage is literally what the dashboard ran on today.
- **"Accuracy of the stats?"** Human-eval harness is in the repo
  (vision/stats/eval_human.py); shot events carry confirmed flags; QA
  page forces re-checks when the system's call changes. We ship measured
  numbers, not vibes.
- **"Why will you beat Pivo?"** Pivo sells a gimbal. We sell the game:
  capture is the cheapest part of the stack, the stats layer is the
  moat, and our capture is 10x cheaper than theirs anyway.
- **"What's next?"** Hoop-class fine-tune → automatic made/missed →
  rebounds → assists; WHOOP-calibrated fatigue analytics; auto-scorekeeping
  with human veto for volunteer-run leagues.

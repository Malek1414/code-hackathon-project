import { COMPARE, type Verdict } from './data/compare'

const WINNER: Record<Verdict, string> = {
  quest: 'Quest 3',
  rayneo: 'X3 Pro',
  even: 'Neither',
}

export function Compare() {
  return (
    <div className="compare">
      <div className="compare-intro">
        <h1 className="compare-title">What survives the move to glasses</h1>
        <p>
          The Quest 3 is the hackathon device and the RayNeo X3 Pro is the roadmap device. They are
          not two versions of the same thing: one shows the wearer a video of the room and can edit
          it, the other paints light on top of a room it cannot touch. Every row below is the same
          question asked of both — and the column that matters is the last one.
        </p>
      </div>

      <table className="compare-table">
        <thead>
          <tr>
            <th>Subsystem</th>
            <th>Property</th>
            <th>Meta Quest 3</th>
            <th>RayNeo X3 Pro</th>
            <th>Better here</th>
          </tr>
        </thead>
        <tbody>
          {COMPARE.map((r) => (
            <tr key={r.property}>
              <td className="c-sys">{r.system}</td>
              <td className="c-prop">
                {r.property}
                <span className="c-sowhat">{r.soWhat}</span>
              </td>
              <td className="c-val">{r.quest}</td>
              <td className="c-val">{r.rayneo}</td>
              <td><span className="verdict" data-v={r.verdict}>{WINNER[r.verdict]}</span></td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="compare-note">
        <h2 className="block-head">On the photosensitivity warning</h2>
        <p>
          Both devices ship with a photosensitive-seizure warning, and on retail hardware it is not
          removable. It is also not the obstacle it looks like. The warning is precautionary and it
          covers content as much as hardware — a systematic review of VR in this population concluded
          that the risk is minimal when the device is used appropriately, and that a history of
          photosensitivity should not be an absolute contraindication.
        </p>
        <p>
          So the useful question is not how to get past the warning. It is which components put
          flicker into the eye, and which ones you control. On the Quest that chain is short and every
          link has a lever: refresh rate is requested by the app and can be held anywhere from 72 to
          207 Hz, peak brightness is capped around 100 nits by the pancake optics, and the whole view
          is a render target a shader can clamp before the wearer sees it. Open the <strong>Flicker
          chain</strong> view on either device to see the components involved.
        </p>
        <p>
          For a clinician the warning is an asset, not a problem: it is evidence the display stack is
          understood and that the device is not being presented as risk-free. The thing worth walking
          into Charité with is the measured refresh rate, the measured luminance, and the filter
          behaviour — not a headset with its warning screen patched out.
        </p>
      </div>
    </div>
  )
}

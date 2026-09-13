import { useState } from 'react'
import { DEVICES } from './data/devices'
import { DeviceAtlas } from './DeviceAtlas'
import { Compare } from './Compare'

export function App() {
  const [tab, setTab] = useState<string>(DEVICES[0].id)
  const device = DEVICES.find((d) => d.id === tab)

  return (
    <div className="app">
      <header className="masthead">
        <span className="masthead-title">Atlas</span>
        <nav className="tabs">
          {DEVICES.map((d) => (
            <button key={d.id} className="tab" data-on={tab === d.id} onClick={() => setTab(d.id)}>
              {d.short}
            </button>
          ))}
          <button className="tab" data-on={tab === 'compare'} onClick={() => setTab('compare')}>
            Quest 3 vs X3 Pro
          </button>
        </nav>
        <span className="masthead-spacer" />
        {device && <span className="masthead-sub">{device.name}</span>}
      </header>

      {device
        ? <DeviceAtlas key={device.id} device={device} />
        : <Compare />}
    </div>
  )
}

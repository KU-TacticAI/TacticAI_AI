import React from 'react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import Game from './game/Game'

// Helper to create paired states for two players in the same row.
// Both sides share the same `sessionId`, `player_names`, and `gameType`.
// `player_names` use the form `test-{userid}` and `player` is set to the userid string.
const makePair = (
  sessionId: string,
  gameType: string,
  leftId: string,
  rightId: string
) => {
  const player_names = [`test-${leftId}`, `test-${rightId}`]
  const leftState = { sessionId, player_names, player: leftId, gameType }
  const rightState = { sessionId, player_names, player: rightId, gameType }
  return [leftState, rightState] as const
}

// Default counts per game type (start at 0)
const defaultCounts = { othello: 0, tictactoe: 0, omok: 0, chess: 0 }

export default function App() {
  const [counts, setCounts] = React.useState(defaultCounts)
  const [sessions, setSessions] = React.useState<Array<{ sessionId: string; gameType: string; leftId: string; rightId: string }>>(() => {
    // start with no sessions when defaults are zero
    return []
  })

  const handleCountChange = (game: string, value: number) => {
    setCounts(prev => ({ ...prev, [game]: Math.max(0, Math.floor(value)) }))
  }

  const generate = () => {
    const newSessions: Array<{ sessionId: string; gameType: string; leftId: string; rightId: string }> = []
    let uid = 1
    // base timestamp string (milliseconds) to prefix session ids
    const baseTs = String(Date.now())
    let sid = 1
    Object.entries(counts).forEach(([gameType, count]) => {
      for (let i = 0; i < (count as number); i++) {
        const leftId = String(uid++)
        const rightId = String(uid++)
        // sessionId is timestamp + sequence number, e.g. "16383091234561", only digits
        const sessionId = baseTs + String(sid++)
        newSessions.push({ sessionId, gameType, leftId, rightId })
      }
    })
    setSessions(newSessions)
  }

  const reset = () => {
    setCounts(defaultCounts)
    setSessions([])
  }

  // start statuses for sessions keyed by sessionId
  const [startStatuses, setStartStatuses] = React.useState<Record<string, string>>({})
  const [starting, setStarting] = React.useState(false)

  const START_URL = '/ai/game-request'

  const modelUrlsByGame: Record<string, string[]> = {
    chess: [
      'https://www.dropbox.com/scl/fi/lv93swyebwkxv81ex0xve/chess_jit_1.pt?rlkey=x7d0c9h0i168eko3k0lk1ldk8&st=vnvobfmd&dl=1',
      'https://www.dropbox.com/scl/fi/lv93swyebwkxv81ex0xve/chess_jit_1.pt?rlkey=x7d0c9h0i168eko3k0lk1ldk8&st=vnvobfmd&dl=1'
    ],
    omok: [
      'https://www.dropbox.com/scl/fi/jco3pz5juzhnxdqc0jzj5/omok_jit_1.pt?rlkey=iwvqc8tum4cjisgg6khy5izrr&st=ebg2xqck&dl=1',
      'https://www.dropbox.com/scl/fi/jco3pz5juzhnxdqc0jzj5/omok_jit_1.pt?rlkey=iwvqc8tum4cjisgg6khy5izrr&st=ebg2xqck&dl=1'
    ],
    othello: [
      'https://www.dropbox.com/scl/fi/w2x1l8h3mly1co8k6ubmc/othello_jit_1.pt?rlkey=kudexue233q45bzig9gcwb5sm&st=1g6xe5c3&dl=1',
      'https://www.dropbox.com/scl/fi/w2x1l8h3mly1co8k6ubmc/othello_jit_1.pt?rlkey=kudexue233q45bzig9gcwb5sm&st=1g6xe5c3&dl=1'
    ],
    tictactoe: [
      'https://www.dropbox.com/scl/fi/ajc67z97kcqmljyk5gyqx/tictactoe_jit_1.pt?rlkey=tqb60ac4d86vi1e74xsnkg1rt&st=5mvo36j6&dl=1',
      'https://www.dropbox.com/scl/fi/ajc67z97kcqmljyk5gyqx/tictactoe_jit_1.pt?rlkey=tqb60ac4d86vi1e74xsnkg1rt&st=5mvo36j6&dl=1'
    ],
  }

  const startAll = async () => {
    if (sessions.length === 0) return
    setStarting(true)
    for (const s of sessions) {
      const { sessionId, gameType, leftId, rightId } = s
      // show pending
      setStartStatuses(prev => ({ ...prev, [sessionId]: 'pending' }))
      try {
        const body = {
          game_id: sessionId,
          game_type: gameType,
          ai_model_ids: ['1','2'],
          ai_model_urls: modelUrlsByGame[gameType.toLowerCase()] || [],
          player_ids: [leftId, rightId]
        }
        const resp = await fetch(START_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        })
        if (!resp.ok) {
          setStartStatuses(prev => ({ ...prev, [sessionId]: `error: ${resp.status}` }))
        } else {
          setStartStatuses(prev => ({ ...prev, [sessionId]: 'started' }))
        }
      } catch (err:any) {
        setStartStatuses(prev => ({ ...prev, [sessionId]: `error` }))
      }
      // small delay between requests
      await new Promise(r => setTimeout(r, 150))
    }
    setStarting(false)
  }

  // Listen for gameFinished events from Game components and mark status finished
  React.useEffect(() => {
    const handler = (e: Event) => {
      const ce = e as CustomEvent<{ sessionId: string }>
      const sid = ce?.detail?.sessionId
      if (sid) setStartStatuses(prev => ({ ...prev, [sid]: 'finished' }))
    }
    window.addEventListener('gameFinished', handler as EventListener)
    return () => window.removeEventListener('gameFinished', handler as EventListener)
  }, [])

  return (
    <div>
      <div className="controls">
        <div className="controls-left">
          <div className="controls-row">
            <label>Othello: <input type="number" min={0} value={counts.othello} onChange={e => handleCountChange('othello', Number(e.target.value))} /></label>
            <label>TicTacToe: <input type="number" min={0} value={counts.tictactoe} onChange={e => handleCountChange('tictactoe', Number(e.target.value))} /></label>
            <label>Omok: <input type="number" min={0} value={counts.omok} onChange={e => handleCountChange('omok', Number(e.target.value))} /></label>
            <label>Chess: <input type="number" min={0} value={counts.chess} onChange={e => handleCountChange('chess', Number(e.target.value))} /></label>
          </div>
          <div className="controls-row">
            <button onClick={generate} disabled={starting}>Generate</button>
            <button onClick={reset} disabled={starting}>Reset</button>
            <button onClick={startAll} disabled={starting || sessions.length===0}>Start</button>
            {starting ? <span style={{marginLeft:8}}>Starting...</span> : null}
          </div>
        </div>

        <div className="controls-right">
          {/* per-game idle/start/finish counts */}
          {(() => {
            // build map from gameType -> counts
            const map: Record<string, { idle: number; start: number; finish: number }> = {}
            // initialize from known game keys (counts keys)
            Object.keys(counts).forEach(k => { map[k] = { idle: 0, start: 0, finish: 0 } })
            sessions.forEach(s => {
              const g = s.gameType
              if (!map[g]) map[g] = { idle: 0, start: 0, finish: 0 }
              const st = startStatuses[s.sessionId] || 'idle'
              if (st === 'finished') map[g].finish++
              else if (st === 'started' || st === 'pending') map[g].start++
              else map[g].idle++
            })
            return Object.entries(map).map(([game, c]) => (
              <div className="controls-game" key={game}>
                <div className="controls-game-name">{game}</div>
                <div className="controls-game-badges">
                  <span className="summary-badge">idle {c.idle}</span>
                  <span className="summary-badge">start {c.start}</span>
                  <span className="summary-badge">finish {c.finish}</span>
                </div>
              </div>
            ))
          })()}
        </div>
      </div>

      <div className="rows">
        {sessions.map(({ sessionId, gameType, leftId, rightId }) => {
          const [leftState, rightState] = makePair(sessionId, gameType, leftId, rightId)
          return (
            <div className="two-column" key={sessionId}>
                <div className="panel">
                  <div className="panel-header">
                    <div className="sid">Session: {sessionId}</div>
                    <div className={`status ${startStatuses[sessionId] || ''}`}>{startStatuses[sessionId] || 'idle'}</div>
                  </div>
                  <MemoryRouter initialEntries={[{ pathname: '/', state: leftState }]}> 
                    <Routes>
                      <Route path="/" element={<Game {...leftState} />} />
                    </Routes>
                  </MemoryRouter>
                </div>

              <div className="panel">
                <div className="panel-header">
                  <div className="sid">Session: {sessionId}</div>
                  <div className={`status ${startStatuses[sessionId] || ''}`}>{startStatuses[sessionId] || 'idle'}</div>
                </div>
                <MemoryRouter initialEntries={[{ pathname: '/', state: rightState }]}> 
                  <Routes>
                    <Route path="/" element={<Game {...rightState} />} />
                  </Routes>
                </MemoryRouter>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

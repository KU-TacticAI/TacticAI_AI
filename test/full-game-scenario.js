// k6-test/full-game-scenario.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { SharedArray } from 'k6/data';
import { Trend } from 'k6/metrics';

// --- Test 옵션 ---
// This scenario runs a full game, which involves multiple requests.
const  MAXVIRTUSERS = 10;
export const options = {
  // 시나리오 설정: 'ramping-vus'를 사용하여 점진적으로 부하를 조절합니다.
  scenarios: {
    full_game_scenario: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: MAXVIRTUSERS }, // 부하증가구간: VUs를 0에서 MAXVUs까지 서서히 늘립니다.
        { duration: '6m', target: MAXVIRTUSERS }, // 최대부하구간: MAXVUs를 계속 유지합니다.
        { duration: '2m', target: 0 },                              // 부하감소구간: VUs를 다시 0으로 서서히 줄입니다.
      ],
      exec: 'fullGameScenario',
      gracefulStop: '1m', //잔여 게임 종료대기(1분)
    },
  },

  // 테스트 성공/실패 기준선 설정
  thresholds: {
    'http_req_duration{name:01_GameRequest}': ['p(95)<1000'],
    'http_req_duration{name:02_PollProgress}': ['p(95)<500'],
    'game_duration': ['p(95)<30000'],
    'http_req_failed': ['rate<0.001'],
    'checks': ['rate>0.99'],
  },

  // 리포트 상세 설정
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'count'],
};

// --- Custom Trend Metric ---
const gameDuration = new Trend('game_duration', true);

// --- 게임종류
const gameData = new SharedArray('gameData', function () {
    return [
        { gameId: 'tictactoe-game', gameType: 'tictactoe', aiModelIds: ['1', '2'], aiModelUrls: ['local', 'local'], playerIds: ['player-1', 'player-2'] },
        { gameId: 'omok-game', gameType: 'omok', aiModelIds: ['1', '2'], aiModelUrls: ['local', 'local'], playerIds: ['player-3', 'player-4'] },
        { gameId: 'chess-game', gameType: 'chess', aiModelIds: ['1', '2'], aiModelUrls: ['local', 'local'], playerIds: ['player-5', 'player-6'] },
        { gameId: 'othello-game', gameType: 'othello', aiModelIds: ['1', '2'], aiModelUrls: ['local', 'local'], playerIds: ['player-7', 'player-8'] },
    ];
});

// --- Target URL ---
const GATEWAY_URL = 'http://15.164.21.224:30080';

// --- Full Game Flow Scenario ---
export function fullGameScenario() {
  const startTime = Date.now();

  // --- Step 1: Start a new game ---
  // Select a random game from the gameData array for each iteration.
  const game = gameData[Math.floor(Math.random() * gameData.length)];
  const gameType = game.gameType; // Get the game type for tagging.
  const uniqueGameId = `${game.gameId}-${__VU}-${__ITER}`;
  const players = game.playerIds;
  const pollingPlayerId = players[0]; // The player ID used to poll for progress.

  const requestPayload = JSON.stringify({
    game_id: uniqueGameId,
    game_type: gameType,
    ai_model_ids: game.aiModelIds,
    ai_model_urls: game.aiModelUrls,
    player_ids: players,
  });

  const requestParams = {
    headers: { 'Content-Type': 'application/json' },
    tags: { 
      name: '01_GameRequest',
      gameType: gameType, // Add gameType tag
    },
  };

  const gameRequestRes = http.post(`${GATEWAY_URL}/game-request`, requestPayload, requestParams);

  if (!check(gameRequestRes, { 'Step 1: Game Request successful': (r) => r.status === 200 })) {
    // If the game fails to start, abort the scenario for this iteration.
    return;
  }

  // --- Step 2: Poll for game progress until it's finished ---
  let isFinished = false;
  let winner = null;
  const maxPolls = 15; // A safeguard against infinite loops.

  for (let i = 0; i < maxPolls; i++) {
    sleep(2); // Wait 2 seconds to simulate AI thinking time and server processing.

    const progressUrl = `${GATEWAY_URL}/ai/progress/${uniqueGameId}/${pollingPlayerId}?n=10`;
    const progressRes = http.get(progressUrl, { 
      tags: { 
        name: '02_PollProgress',
        gameType: gameType, // Add gameType tag
      }, 
    });

    if (!check(progressRes, { 'Polling successful': (r) => r.status === 200 })) {
      // If polling fails, stop trying for this game.
      break;
    }

    try {
      const progressMessages = progressRes.json();
      if (Array.isArray(progressMessages) && progressMessages.length > 0) {
        // Check the very last message in the array.
        const lastMessage = progressMessages[progressMessages.length - 1];
        
        if (lastMessage && lastMessage.is_finished) {
          isFinished = true;
          winner = lastMessage.winner;
          console.log(`Game ${uniqueGameId} (${gameType}) finished. Winner: ${winner || 'Draw'}`);
          break; // Exit the loop as the game is over.
        }
      }
    } catch (e) {
      console.error(`Error parsing JSON for game ${uniqueGameId}: ${e.message}`);
      break;
    }
  }

  // --- Step 3: Final check and record metrics ---
  check(null, {
    'Step 3: Game successfully finished': () => isFinished,
  });

  // If the game finished, record its total duration with the gameType tag.
  if (isFinished) {
    const endTime = Date.now();
    gameDuration.add(endTime - startTime, { gameType: gameType });
  }
}

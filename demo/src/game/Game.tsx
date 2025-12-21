import React from "react";
import TicTacToe from "./TicTacToe";
import Othello from "./Othello";
import Chess from "./Chess";
import Omok from "./Omok";
import {useLocation} from "react-router-dom";
// 다른 게임 컴포넌트들도 필요시 import
// import Checkers from "./Checkers";

// Use relative API paths so the dev server (or the same origin) can proxy requests
// This avoids mixed-content when the page is served over HTTPS but backend is HTTP.
const API_PROGRESS_URL = `/ai/progress`;
const API_START_URL = `/ai/game-request/`;

interface GameProps {
  gameType: string;
  sessionId: string;
  player_names: string[]; // 플레이어 이름이 필요할 경우 추가
  player: string;
}

const Game: React.FC<GameProps> = () => {
  const location = useLocation();
  const { sessionId, player_names, player, gameType } = location.state as {
    sessionId: string;
    player_names: string[];
    player: string;
    gameType: string;
  };
  // console.log('Game 컴포넌트로 전달된 gameType:', gameType);
  // 게임 진행상황 저장하기 위한 배열 useState
  const [gameProgress, setGameProgress] = React.useState<any[]>([]);
  // 현제 인덱스
  const [currentIndex, setCurrentIndex] = React.useState(-1);

  const progressRef = React.useRef(gameProgress);
  const indexRef    = React.useRef(currentIndex);
  const isMountedRef = React.useRef(true);
  const fetchTimeoutRef = React.useRef<number | null>(null);
  const abortControllerRef = React.useRef<AbortController | null>(null);
  const finishedRef = React.useRef(false);
  // 게임 타입에 따라 렌더링할 컴포넌트 선택
  const renderGame = () => {
    switch (gameType.toLowerCase()) {
      case "오셀로":
      case "othello":
        if (gameProgress.length === 0) {
          return <Othello />; // 게임 진행 정보가 없을 때
        }
        return <Othello gameData={gameProgress[currentIndex]} />;
      case "틱택토":
      case "tictactoe":
        if (gameProgress.length === 0) {
          return <TicTacToe />; // 게임 진행 정보가 없을 때
        }
        return <TicTacToe gameData={gameProgress[currentIndex]} />;      
      case "체스":
      case "chess":
        if (gameProgress.length === 0 || currentIndex < 0 || currentIndex >= gameProgress.length) {
          return <Chess />; // 게임 진행 정보가 없을 때
        }
        return <Chess gameData={gameProgress[currentIndex]} />;
      case "오목":
      case "omok":
        if (gameProgress.length === 0) {
          return <Omok />; // 게임 진행 정보가 없을 때
        }
        return <Omok gameData={gameProgress[currentIndex]} />;
      default:
        return <div>지원하지 않는 게임 타입입니다: {gameType}</div>;
    }
  };

  // 게임진행정보 받아오는 함수
  const fetchGameProgress = async () => {
    let is_finished = false;
    try {
      // abort previous fetch if any
      if (abortControllerRef.current) {
        try { abortControllerRef.current.abort(); } catch {};
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;
      const response = await fetch(`${API_PROGRESS_URL}/${sessionId}/${player}`, { signal: controller.signal });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      // console.log('게임 진행 정보:', data);
      // 데이터가 비어있는경우
      if (Object.keys(data).length === 0) {
      }else {
        // 게임 진행 정보가 있는 경우 처리
        // 마지막 정보 가져오기
        // data에 player_names 추가
        data.forEach((item: any) => {
          item.player_names = player_names; // player_names 추가
        });
        const lastInfo = data[data.length - 1];
        // console.log('마지막 게임 진행 정보:', lastInfo);
        if (lastInfo.is_finished) {
          is_finished = true; // 게임이 종료되었음을 표시
          // dispatch a window event once to notify parent that this session finished
          if (!finishedRef.current) {
            try {
              const ev = new CustomEvent('gameFinished', { detail: { sessionId } });
              window.dispatchEvent(ev);
            } catch (e) {}
            finishedRef.current = true;
          }
        }
        else {
        }
      }
      if (isMountedRef.current) {
        setGameProgress(prevProgress => [...prevProgress, ...data]);

        if (!is_finished) {
          // schedule next fetch but keep the timer id so we can clear on unmount
          fetchTimeoutRef.current = window.setTimeout(() => {
            if (isMountedRef.current) fetchGameProgress();
          }, 1000) as unknown as number;
        }
      }

      // 게임 진행 정보 처리 로직 추가
    } catch (error) {
      console.error('게임 진행 정보 가져오기 실패:', error);
    }
  };

  // 게임 보드를 업데이트하는 함수
  const updateGameBoard = React.useCallback(() => {
    setCurrentIndex(prev => {
      const next = prev + 1;
      if (next < gameProgress.length) {
        // console.log('업데이트:', gameProgress[next]);
        return next;
      }
      return prev;
    });
  }, [gameProgress]);

  // 컴포넌트가 마운트될 때 게임 진행 정보 가져오기
  React.useEffect(() => {
    isMountedRef.current = true;
    fetchGameProgress();
    return () => {
      // mark unmounted and cancel pending work
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        try { abortControllerRef.current.abort(); } catch {}
      }
      if (fetchTimeoutRef.current) {
        clearTimeout(fetchTimeoutRef.current);
        fetchTimeoutRef.current = null;
      }
    };
  },[]);
  React.useEffect(() => { progressRef.current = gameProgress; }, [gameProgress]);
  React.useEffect(() => { indexRef.current    = currentIndex; }, [currentIndex]);
  // 컴포넌트가 마운트될 때 게임 보드 업데이트
  React.useEffect(() => {
    const id = setInterval(() => {
      const prog = progressRef.current;
      const idx  = indexRef.current;

      // 새로운 스냅샷이 남아 있으면 하나만 꺼내서 처리
      if (idx < prog.length - 1) {
        setCurrentIndex(idx + 1);
        // console.log('보드 업데이트:', prog[idx + 1]);
      }
      // else: 더 이상 처리할 게 없으면 그냥 가만히 둡니다.
    }, 100);

    return () => clearInterval(id);
  }, []);  // 빈 배열 → 마운트 때만 실행

  return (
    <div>
      <div className="game-container">
        {renderGame()}
      </div>
    </div>
  );
};

export default Game;

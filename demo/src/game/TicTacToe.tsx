import React from 'react';
import './TicTacToe.css';
import BaseBoardGame from './BaseBoardGame';
import { GameComponentProps, Score } from './GameTypes';

const TicTacToe: React.FC<GameComponentProps> = ({ gameData }) => {
  // 승리 라인 체크 함수
  const checkWinningLine = (boardState: number[]): number[][] | null => {
    const lines = [
      // 가로선
      [[0,0], [0,1], [0,2]],
      [[1,0], [1,1], [1,2]],
      [[2,0], [2,1], [2,2]],
      // 세로선
      [[0,0], [1,0], [2,0]],
      [[0,1], [1,1], [2,1]],
      [[0,2], [1,2], [2,2]],
      // 대각선
      [[0,0], [1,1], [2,2]],
      [[0,2], [1,1], [2,0]]
    ];

    // 1차원 배열을 2차원으로 변환
    const board: number[][] = [];
    for (let i = 0; i < 3; i++) {
      const row: number[] = [];
      for (let j = 0; j < 3; j++) {
        row.push(boardState[i * 3 + j]);
      }
      board.push(row);
    }

    for (const line of lines) {
      const [a, b, c] = line;
      const valueA = board[a[0]][a[1]];
      const valueB = board[b[0]][b[1]];
      const valueC = board[c[0]][c[1]];
      
      if (valueA !== 0 && valueA === valueB && valueB === valueC) {
        return line;
      }
    }
    
    return null;
  };

  // 셀 렌더링 함수
  const renderCell = (value: number, row: number, col: number, isLastMove: boolean): React.ReactElement => {
    let cellClass = 'tictactoe-cell';
    let content: React.ReactElement | null = null;

    if (value === 1) {
      cellClass += ' x-mark';
      content = <div className="tictactoe-mark tictactoe-x-mark-inner">X</div>;
    } else if (value === -1) {
      cellClass += ' o-mark';
      content = <div className="tictactoe-mark tictactoe-o-mark-inner">O</div>;
    } else {
      cellClass += ' empty-cell';
    }

    if (isLastMove) {
      cellClass += ' last-move';
    }

    // 승리 라인 체크 (gameData가 있을 때만)
    const winningLine = gameData ? checkWinningLine(gameData.board_state) : null;
    const isWinningCell = winningLine?.some(([r, c]) => r === row && c === col);
    if (isWinningCell) {
      cellClass += ' winning-cell';
    }

    return (
      <div key={`${row}-${col}`} className={cellClass}>
        {content}
      </div>
    );
  };

  // 점수 계산
  const calculateScore = (boardState: number[]): Score => {
    const x = boardState.filter(cell => cell === 1).length;
    const o = boardState.filter(cell => cell === -1).length;
    return { x, o };
  };

  return (
    <BaseBoardGame
      gameData={gameData}
      gameType="tictactoe"
      renderCell={renderCell}
      calculateScore={calculateScore}
    >
      {/* 승리 라인 정보 추가 표시 */}
      {gameData && checkWinningLine(gameData.board_state) && (
        <div style={{textAlign: 'center', color: '#27ae60', fontWeight: 'bold', marginBottom: '20px'}}>
          승리 라인 발견!
        </div>
      )}
    </BaseBoardGame>
  );
};

export default TicTacToe;

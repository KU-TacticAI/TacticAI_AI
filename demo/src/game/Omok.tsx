import React from 'react';
import './Omok.css';
import BaseBoardGame from './BaseBoardGame';
import { GameComponentProps, Score, GAME_CONFIGS } from './GameTypes';

const Omok: React.FC<GameComponentProps> = ({ gameData }) => {
  const config = GAME_CONFIGS['omok'];
  const size = config.boardSize;

  // 보드를 2차원 배열로 변환
  const toBoard = (boardState: number[]) => {
    const board: number[][] = [];
    for (let r = 0; r < size; r++) {
      const row: number[] = [];
      for (let c = 0; c < size; c++) {
        row.push(boardState[r * size + c]);
      }
      board.push(row);
    }
    return board;
  };

  // 5목 승리 체크 - 승리한 좌표 배열 반환
  const checkWinningLine = (boardState: number[] | undefined): number[][] | null => {
    if (!boardState) return null;
    const board = toBoard(boardState);

    const dirs = [
      [0, 1], // horizontal
      [1, 0], // vertical
      [1, 1], // diag down-right
      [1, -1] // diag down-left
    ];

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const val = board[r][c];
        if (val === 0) continue;

        for (const [dr, dc] of dirs) {
          const coords: number[][] = [[r, c]];
          let rr = r + dr;
          let cc = c + dc;
          while (rr >= 0 && rr < size && cc >= 0 && cc < size && board[rr][cc] === val) {
            coords.push([rr, cc]);
            rr += dr;
            cc += dc;
          }

          // also check backward direction to capture full line
          rr = r - dr;
          cc = c - dc;
          while (rr >= 0 && rr < size && cc >= 0 && cc < size && board[rr][cc] === val) {
            coords.unshift([rr, cc]);
            rr -= dr;
            cc -= dc;
          }

          if (coords.length >= 5) {
            // return only first 5 in sequence (centered)
            return coords.slice(0, 5);
          }
        }
      }
    }

    return null;
  };

  const renderCell = (value: number, row: number, col: number, isLastMove: boolean): React.ReactElement => {
    const winningLine = gameData ? checkWinningLine(gameData.board_state) : null;
    const isWinningCell = winningLine?.some(([r, c]) => r === row && c === col);

    let className = 'omok-cell';
    if (value === 1) className += ' black-stone';
    if (value === -1) className += ' white-stone';
    if (isLastMove) className += ' last-move';
    if (isWinningCell) className += ' winning-cell';

    return (
      <div key={`${row}-${col}`} className={className}>
        {/* stone visuals handled by CSS */}
      </div>
    );
  };

  const calculateScore = (boardState: number[]): Score => {
    const black = boardState.filter(x => x === 1).length;
    const white = boardState.filter(x => x === -1).length;
    return { black, white } as Score;
  };

  return (
    <BaseBoardGame
      gameData={gameData}
      gameType="omok"
      renderCell={renderCell}
      calculateScore={calculateScore}
    />
  );
};

export default Omok;

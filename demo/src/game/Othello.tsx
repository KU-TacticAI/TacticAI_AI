import React from 'react';
import './Othello.css';
import BaseBoardGame from './BaseBoardGame';
import { GameComponentProps, Score } from './GameTypes';

const Othello: React.FC<GameComponentProps> = ({ gameData }) => {
  // 셀 렌더링 함수
  const renderCell = (value: number, row: number, col: number, isLastMove: boolean): React.ReactElement => {
    let cellClass = 'othello-cell';
    let content: React.ReactElement | null = null;

    if (value === -1) {
      cellClass += ' white-piece';
      content = <div className="othello-piece othello-white-piece-inner"></div>;
    } else if (value === 1) {
      cellClass += ' black-piece';
      content = <div className="othello-piece othello-black-piece-inner"></div>;
    } else {
      cellClass += ' empty-cell';
    }

    if (isLastMove) {
      cellClass += ' last-move';
    }

    return (
      <div key={`${row}-${col}`} className={cellClass}>
        {content}
      </div>
    );
  };

  // 점수 계산
  const calculateScore = (boardState: number[]): Score => {
    const white = boardState.filter(cell => cell === -1).length;
    const black = boardState.filter(cell => cell === 1).length;
    return { white, black };
  };

  return (
    <BaseBoardGame
      gameData={gameData}
      gameType="othello"
      renderCell={renderCell}
      calculateScore={calculateScore}
    />
  );
};

export default Othello;

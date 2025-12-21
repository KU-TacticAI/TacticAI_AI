import React from 'react';
import './Chess.css';
import BaseBoardGame from './BaseBoardGame';
import { GameComponentProps, Score, ChessPiece } from './GameTypes';

const Chess: React.FC<GameComponentProps> = ({ gameData }) => {
  // 체스 기물 유니코드 심볼
  const getPieceSymbol = (piece: ChessPiece): string => {
    const symbols = {
      white: {
        king: '♔',
        queen: '♕',
        rook: '♖',
        bishop: '♗',
        knight: '♘',
        pawn: '♙'
      },
      black: {
        king: '♚',
        queen: '♛',
        rook: '♜',
        bishop: '♝',
        knight: '♞',
        pawn: '♟'
      }
    };
    return symbols[piece.color][piece.type];
  };  // 기물 값으로 ChessPiece 객체 반환
  const getPieceFromValue = (value: number): ChessPiece | null => {
    // value가 undefined이거나 null이거나 0인 경우 null 반환
    if (value == null || value === 0) return null;
    
    // PIECE_NAMES 매핑에 따라 수정
    const pieceMap: { [key: string]: ChessPiece } = {
      "1": { type: 'pawn', color: 'white' },      // 백색 폰
      "2": { type: 'knight', color: 'white' },    // 백색 나이트
      "3": { type: 'bishop', color: 'white' },    // 백색 비숍
      "4": { type: 'rook', color: 'white' },      // 백색 룩
      "5": { type: 'queen', color: 'white' },     // 백색 퀸
      "6": { type: 'king', color: 'white' },      // 백색 킹
      "-1": { type: 'pawn', color: 'black' },     // 흑색 폰
      "-2": { type: 'knight', color: 'black' },   // 흑색 나이트
      "-3": { type: 'bishop', color: 'black' },   // 흑색 비숍
      "-4": { type: 'rook', color: 'black' },     // 흑색 룩
      "-5": { type: 'queen', color: 'black' },    // 흑색 퀸
      "-6": { type: 'king', color: 'black' }      // 흑색 킹
    };
    
    return pieceMap[value.toString()] || null;
  };

  // 셀 색상 결정 (체스판 체크무늬)
  const getCellColor = (row: number, col: number): string => {
    return (row + col) % 2 === 0 ? 'light' : 'dark';
  };
  // 셀 렌더링 함수
  const renderCell = (value: number, row: number, col: number, isLastMove: boolean): React.ReactElement => {
    const cellColor = getCellColor(row, col);
    let cellClass = `chess-cell chess-cell-${cellColor}`;
    let content: React.ReactElement | null = null;

    const piece = getPieceFromValue(value);
    if (piece) {
      const pieceSymbol = getPieceSymbol(piece);
      cellClass += ` chess-piece chess-${piece.color}-piece`;
      content = <div className="chess-piece-symbol">{pieceSymbol}</div>;
    }

    if (isLastMove) {
      cellClass += ' chess-last-move';
    }

    return (
      <div key={`${row}-${col}`} className={cellClass}>
        {content}
      </div>
    );
  };
  // 점수 계산 (기물 가치 합계)
  const calculateScore = (boardState: number[]): Score => {
    const pieceValues: { [key: number]: number } = {
      1: 1,   // 폰
      2: 3,   // 나이트
      3: 3,   // 비숍
      4: 5,   // 룩
      5: 9,   // 퀸
      6: 0    // 킹 (무한대 가치)
    };

    let whiteScore = 0;
    let blackScore = 0;

    boardState.forEach(value => {
      // value가 유효한 숫자인지 확인
      if (typeof value === 'number' && !isNaN(value)) {
        if (value > 0) {
          // 양수는 백색 기물
          whiteScore += pieceValues[value] || 0;
        } else if (value < 0) {
          // 음수는 흑색 기물
          blackScore += pieceValues[-value] || 0;
        }
      }
    });

    return { white: whiteScore, black: blackScore };
  };

  return (
    <BaseBoardGame
      gameData={gameData}
      gameType="chess"
      renderCell={renderCell}
      calculateScore={calculateScore}
    >
      {/* 체스 특별 정보 표시 */}
      {gameData && (
        <div className="chess-game-status">
          {gameData.is_finished && (
            <div className="chess-game-result">
              {gameData.winner ? `승자: ${gameData.winner}` : '무승부'}
            </div>
          )}
        </div>
      )}
    </BaseBoardGame>
  );
};

export default Chess;

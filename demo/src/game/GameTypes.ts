export interface GameRoom {
    roomId: string; // Changed from roomId
    roomName: string;
    gameType: string;
    hostUserId: string;
    hostRanking: number;
    playerCount: number;
    maxPlayers: number;
    isPrivate: boolean;
    status: string;
    isDeleted: boolean;
}

export interface AI {
    aiId: string;
    userId: string;
    name: string;
    description: string;
    gameType: string;
    aiUrl: string;
    aiSize: string;
    score: string;
    tier: string;
    version : string;
    uploadDate: string;
    updateAt: string;
}

export interface Player {
    userId: string;
    nickname: string;
    profileUrl: string;
    ranking: number;
    isReady: boolean;
    selectedAi: AI | null;
}

export interface GameRoomDetail extends GameRoom {
    players: Player[];
    isDeleted: boolean;
}

// 공통 게임 데이터 인터페이스
export interface GameData {
  game_type: string;
  board_state: number[];
  turn_number: number;
  current_turn: number;
  player_ids: string[];
  player_names: string[];
  last_move: string | null;
  is_finished: boolean;
  winner: string | null;
  is_success: boolean;
  timestamp: string;
  avg_response_times?: number[];
}

// 공통 컴포넌트 Props 인터페이스
export interface GameComponentProps {
  gameData?: GameData;
}

// 공통 좌표 타입
export interface Coordinates {
  col: number;
  row: number;
}

// 공통 점수 타입 (제네릭으로 확장 가능)
export interface Score {
  [key: string]: number;
}

// 체스 기물 타입 정의
export interface ChessPiece {
  type: 'king' | 'queen' | 'rook' | 'bishop' | 'knight' | 'pawn';
  color: 'white' | 'black';
}

// 게임 설정 인터페이스
export interface GameConfig {
  boardSize: number; // 보드 크기 (오셀로: 8, 틱택토: 3)
  cellCount: number; // 총 셀 개수 (보드크기^2)
  playerSymbols: string[]; // 플레이어 심볼 ['X', 'O'] 또는 ['●', '○']
  playerColors: string[]; // 플레이어 색상
  defaultBoardState: number[]; // 기본 보드 상태
  isChess?: boolean; // 체스 여부
  chessPieces?: { [key: string]: ChessPiece }; // 체스 기물 매핑
}

// 게임별 설정
export const GAME_CONFIGS: { [key: string]: GameConfig } = {
  othello: {
    boardSize: 8,
    cellCount: 64,
    playerSymbols: ['●', '○'],
    playerColors: ['black', 'white'],
    defaultBoardState: [
      0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,
      0,0,0,-1,1,0,0,0,
      0,0,0,1,-1,0,0,0,
      0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0
    ]
  },  tictactoe: {
    boardSize: 3,
    cellCount: 9,
    playerSymbols: ['X', 'O'],
    playerColors: ['red', 'blue'],
    defaultBoardState: [0,0,0,0,0,0,0,0,0]
  },
  chess: {
    boardSize: 8,
    cellCount: 64,
    playerSymbols: ['♔', '♚'],
    playerColors: ['white', 'black'],    defaultBoardState: [
      4, 2, 3, 5, 6, 3, 2, 4,   // 백색 주요 기물 (위쪽)
      1, 1, 1, 1, 1, 1, 1, 1,   // 백색 폰
      0, 0, 0, 0, 0, 0, 0, 0,   // 빈 칸
      0, 0, 0, 0, 0, 0, 0, 0,   // 빈 칸
      0, 0, 0, 0, 0, 0, 0, 0,   // 빈 칸
      0, 0, 0, 0, 0, 0, 0, 0,   // 빈 칸
      -1,-1,-1,-1,-1,-1,-1,-1,  // 흑색 폰
      -4,-2,-3,-5,-6,-3,-2,-4   // 흑색 주요 기물 (아래쪽)
    ],
    isChess: true,    chessPieces: {
      "1": { type: 'pawn', color: 'white' },
      "2": { type: 'knight', color: 'white' },
      "3": { type: 'bishop', color: 'white' },
      "4": { type: 'rook', color: 'white' },
      "5": { type: 'queen', color: 'white' },
      "6": { type: 'king', color: 'white' },
      "-1": { type: 'pawn', color: 'black' },
      "-2": { type: 'knight', color: 'black' },
      "-3": { type: 'bishop', color: 'black' },
      "-4": { type: 'rook', color: 'black' },
      "-5": { type: 'queen', color: 'black' },
      "-6": { type: 'king', color: 'black' }
    }
  },
  omok: {
    boardSize: 15,
    cellCount: 225,
    playerSymbols: ['●', '○'],
    playerColors: ['black', 'white'],
    defaultBoardState: Array(225).fill(0)
  }
};

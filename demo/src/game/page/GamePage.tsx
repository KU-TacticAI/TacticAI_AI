import Game from "../Game";
import { useEffect, useState } from "react";
import { getGameLobbyApi } from "../../../api/Api";
import Layout from "../../layout/Layout";
import "./GamePage.css";

interface GameProps {
  gameType: string;
  sessionId: string;
  player_names: string[];
  player_id: string;
}

const GamePage = () => {
  const [gameProbs, setGameProbs] = useState<GameProps | null>(null);

  useEffect(() => {
    const fetchGameLobby = async () => {
      try {
        const roomId = '1';

        // const response = await getGameLobbyApi(roomId);

        // setGameProbs({
        //   gameType: response.data.gameType,
        //   sessionId: response.data.sessionId,
        //   player_names: response.data.player_names,
        //   player_id: response.data.player_id,
        // });

        setGameProbs({
          gameType: 'chess',
          sessionId: '1', // 서버 응답에 맞춰 수정
          player_names: ['test1', 'test2'],
          player_id: '1', // 서버 응답에 맞춰 수정
        });

      } catch (error) {
        console.error("Failed to fetch game lobby data:", error);
      }
    };

    fetchGameLobby();
  }, []);

  if (!gameProbs) {
    return <div>게임 데이터를 불러오는 중...</div>;
  }

  return (
      <Layout>
        <div className='form-container'>
          <Game
              gameType={gameProbs.gameType}
              sessionId={gameProbs.sessionId}
              player_names={gameProbs.player_names}
              player={gameProbs.player_id}
          />
        </div>
      </Layout>
  );
};

export default GamePage;

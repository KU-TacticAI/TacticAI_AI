FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# 작업 디렉토리 설정
WORKDIR /app

# PyTorch 이미지에는 이미 필요한 컴파일러가 포함되어 있음
# 추가 패키지가 필요한 경우에만 설치

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 모델 파일들이 포함된 models 디렉토리 복사
# (모델 파일들이 이미 포함되어 있음)

# 포트 노출 (필요한 경우)
EXPOSE 8001

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 애플리케이션 실행
CMD ["python", "main.py"]
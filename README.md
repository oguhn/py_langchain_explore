# Document - GPT

업로드한 문서기반으로 질문에 대한 답변을 제공합니다.

사전준비
1. OpenAI API 키를 준비합니다.
2. .txt파일을 준비합니다. (테스트용으로 35~70kb정도가 적당 -> 코드 줄 1200줄 이하정도)
    o4-mini-2024-07-18 모델기준 36번 요청 기준 0.01 달러 소요 -> 한 파일당 임베딩 요청이 여러번 있으니 참고!

### Install 
```bash
uv install
```

### Usage
```bash
uv run streamlit run app.py
```

### History
#### 2025-05-02
    - 다른 페이지 삭제 -> Only DocumentGPT Page 유지.
    - OpenAI API 키를 입력받는 함수 추가.


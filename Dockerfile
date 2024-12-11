FROM python:3.9-slim

# Установить зависимости
WORKDIR /ml
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Скопировать код приложения
COPY . .

# Указать порт
ENV PORT=8080

# Команда запуска
CMD streamlit run main.py --server.port=$PORT --server.enableCORS=false

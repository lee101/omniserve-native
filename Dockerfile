FROM debian:bookworm-slim AS build

RUN apt-get update && apt-get install --yes --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Makefile ./
COPY src ./src
COPY models ./models
RUN make all

FROM debian:bookworm-slim

RUN apt-get update && apt-get install --yes --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/omniserve /usr/local/bin/omniserve
COPY --from=build /src/models /app/models
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/omniserve"]
CMD ["--models", "/app/models/models.csv", "--listen", "0.0.0.0"]

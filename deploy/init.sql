-- Runs once on first start of the timescaledb volume. Change → `make down VOLUMES=1`.
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE readings (
    ts        timestamptz NOT NULL,
    meter_id  text        NOT NULL,
    kw        double precision,
    PRIMARY KEY (ts, meter_id)
);
SELECT create_hypertable('readings', 'ts');

CREATE TABLE weather_obs (
    ts           timestamptz NOT NULL,
    station_id   text        NOT NULL,
    temp_c       double precision,
    dewpoint_c   double precision,
    rh           double precision,
    wind_kph     double precision,
    gust_kph     double precision,
    pressure_hpa double precision,
    precip_mm    double precision,
    PRIMARY KEY (ts, station_id)
);
SELECT create_hypertable('weather_obs', 'ts');

CREATE TABLE scores (
    ts            timestamptz NOT NULL,
    meter_id      text        NOT NULL,
    model_version text,
    actual_kw     double precision,
    expected_kw   double precision,
    lower_kw      double precision,
    upper_kw      double precision,
    z             double precision,
    cusum_pos     double precision,
    cusum_neg     double precision,
    excess_kwh    double precision,
    alert_type    text,
    weather_stale boolean,
    latency_ms    double precision,
    PRIMARY KEY (ts, meter_id)
);
SELECT create_hypertable('scores', 'ts');

CREATE TABLE alerts (
    id            bigserial PRIMARY KEY,
    meter_id      text        NOT NULL,
    alert_type    text        NOT NULL,
    started_at    timestamptz NOT NULL,
    ended_at      timestamptz,
    peak_z        double precision,
    excess_kwh    double precision,
    co2_kg        double precision,
    cost_usd      double precision,
    observed_kw   double precision,
    expected_kw   double precision,
    drivers       jsonb,
    model_version text,
    acknowledged  boolean NOT NULL DEFAULT false
);
CREATE INDEX alerts_open_idx ON alerts (meter_id) WHERE ended_at IS NULL;


-- MLflow keeps its own database on the same server
CREATE DATABASE mlflow;

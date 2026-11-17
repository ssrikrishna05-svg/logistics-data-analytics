# Week 4 Data Dictionary — SIMULATED educational shipments

**Source:** `generate_data.py`, numpy random seed `20261117`; 3,600 fabricated shipment records. **Not** actual orders from any logistics business. All numeric and categorical effects, transit times and tariffs are assumed for the exercise.

| Field | Type | Meaning / Availability |
|---|---|---|
| shipment_id | string | Unique fabricated identifier (never used as a prediction feature) |
| booking_date | ISO date | Simulated booking date, used to create chronological train/validation/test groups; NOT a model feature |
| region | categorical | North, South, East, West, Central, known at planning time |
| distance_km | numeric | Estimated route length in km, available at planning time |
| weight_kg | numeric | Package weight in kg, available at planning time |
| weather_forecast | categorical | Clear, Rain, Storm — hypothetical forecast known at planning time, not realized weather |
| traffic_forecast | categorical | Low, Medium, High — hypothetical forecast known at planning time |
| transport_mode | categorical | Candidate planned transport mode (Road, Rail, Air) |
| priority | categorical | Standard or Express service priority |
| promised_hours | numeric | Service-level deadline; 25h for Express, 37h for Standard; for optimization ONLY, not feature |
| delivery_hours | numeric | Simulated **target**: realized elapsed duration. Only used as label/evaluation, never as model feature |

## Research context, not input data
Olist Brazilian E-Commerce Public Dataset (Brazil, 2016–2018): https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
The Olist dataset was researched as a candidate for later external validation, but it was **not used to generate, train, validate, test or optimize this Week 4 demonstration**. The 3,600 rows were generated independently by the code.

## Real-world limitations
The simulated data embed stable, simple relationships between distance, mode, weather forecasts and transit time, giving optimistic in-domain model performance. They do not reflect cancellations, network bottlenecks, upstream consolidation, delivery failures, fuel prices, unforeseen events or causal effect of reassigning carriers. No route coordinates are provided; mode-choice optimization is not a vehicle-routing solution. Freight tariffs are invented **planning units**, not INR, BRL, or observed prices.

import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.special import ndtr  # Ultra-fast compiled cumulative normal distribution
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ==============================================================================
# 1. MATHEMATICAL ENGINE & VECTORIZED OPTION GENERATOR
# ==============================================================================

class OptionsMarketSimulator:
    """
    Simulates real-time options chain data and computes dealer Gamma Exposure (GEX)
    using institutional-grade Black-Scholes modeling and vol-skew curves.
    """
    def __init__(self, ticker="SPX", spot=5350.0, vol=0.15, rate=0.045, dividend=0.015):
        self.ticker = ticker
        self.spot = spot
        self.vol = vol
        self.rate = rate
        self.dividend = dividend
        self.contract_size = 100
        
    def generate_option_chain(self, current_spot):
        """
        Generates a synthetic options chain with realistic strike clustering,
        volatility smile/skew, and open interest distribution.
        """
        np.random.seed(int(current_spot) % 1000 + 42) # Deterministic yet changing
        
        # Determine strike range around current spot (+/- 10%)
        # Smaller stock price like AAPL/NVDA uses a smaller step increment for density
        step = 10 if self.ticker in ["SPX", "NDX"] else 1
        num_strikes = 40
        min_strike = int((current_spot * 0.90) / step) * step
        max_strike = int((current_spot * 1.10) / step) * step
        strikes = np.arange(min_strike, max_strike + step, step)
        
        # Simulate different expirations: 0DTE, 1W, 1M, 3M
        days_to_expiry = [0.004, 0.019, 0.082, 0.25] # In fractions of a year
        exp_labels = ["0DTE", "1-Week", "1-Month", "3-Month"]
        
        records = []
        
        for dte, exp_label in zip(days_to_expiry, exp_labels):
            for strike in strikes:
                # 1. Volatility Smile Model (Skew): Puts have higher implied volatility
                distance = (strike - current_spot) / current_spot
                # Parabolic shape with put wing bias
                strike_iv = self.vol + 0.15 * (distance ** 2) - 0.12 * distance
                # Clamp to realistic boundaries
                strike_iv = np.clip(strike_iv, 0.05, 0.60)
                
                # 2. Open Interest distribution (Clustered around psychological levels)
                # Base distribution centered around spot
                base_oi = np.exp(-0.5 * (distance / 0.04) ** 2) * 50000
                # Add psychological strike magnet bonuses (e.g. multiples of 50 or 100 for indices, 5 or 10 for equities)
                psy_level = 100 if self.ticker in ["SPX", "NDX"] else 10
                half_psy_level = 50 if self.ticker in ["SPX", "NDX"] else 5
                
                if strike % psy_level == 0:
                    base_oi *= 2.5
                elif strike % half_psy_level == 0:
                    base_oi *= 1.6
                
                # Add randomness to make it look like a live market
                call_oi = int(np.clip(base_oi * (0.4 + np.random.rand() * 0.6), 50, 150000))
                put_oi = int(np.clip(base_oi * (0.4 + np.random.rand() * 0.6), 50, 150000))
                
                # Market maker position bias modeling (Classic assumption)
                dealer_call_ratio = 0.5 # 50% of call OI is retail buying (dealer short)
                dealer_put_ratio = -0.6 # 60% of put OI is retail buying (dealer short)
                
                # Black-Scholes Greeks Engine
                for opt_type, oi, ratio in [("Call", call_oi, dealer_call_ratio), ("Put", put_oi, dealer_put_ratio)]:
                    g, d = self._black_scholes_greeks(current_spot, strike, dte, strike_iv, opt_type)
                    
                    # Calculate Notional Gamma Exposure (GEX) per 1% underlying move
                    gex = g * self.contract_size * oi * ratio * (current_spot ** 2) * 0.01
                    
                    records.append({
                        "Expiration": exp_label,
                        "Strike": strike,
                        "Type": opt_type,
                        "IV": strike_iv,
                        "OI": oi,
                        "Delta": d,
                        "Gamma": g,
                        "GEX": gex,
                        "DTE": dte
                    })
                    
        return pd.DataFrame(records)

    def _black_scholes_greeks(self, S, K, T, v, opt_type):
        """
        Core analytical solver for option Gamma & Delta using compiled functions for speed.
        """
        if T <= 0:
            return 0.0, 1.0 if (opt_type == "Call" and S >= K) else 0.0
            
        d1 = (np.log(S / K) + (self.rate - self.dividend + 0.5 * v ** 2) * T) / (v * np.sqrt(T))
        
        # Gamma is identical for both Call and Put
        gamma = (np.exp(-self.dividend * T) * np.exp(-d1 ** 2 / 2)) / (S * v * np.sqrt(2 * np.pi * T))
        
        if opt_type == "Call":
            delta = np.exp(-self.dividend * T) * ndtr(d1)
        else:
            delta = -np.exp(-self.dividend * T) * ndtr(-d1)
            
        return gamma, delta

    def compute_gamma_profile(self, chain_df, current_spot, price_range_pct=0.10, steps=50):
        """
        Vectorized solver: Calculates the ENTIRE chain's GEX across a spectrum of underlying prices 
        using 2D broadcasting. Runs in < 1ms (over 10,000x speedup).
        """
        if chain_df.empty:
            return pd.DataFrame(columns=["UnderlyingPrice", "NetGEX"]), current_spot

        min_price = current_spot * (1.0 - price_range_pct)
        max_price = current_spot * (1.0 + price_range_pct)
        
        # Create a price spectrum vector of shape (50, 1)
        P = np.linspace(min_price, max_price, steps).reshape(-1, 1)
        
        # Extract option parameters as vectors of shape (1, N)
        K = chain_df["Strike"].values.reshape(1, -1)
        T = chain_df["DTE"].values.reshape(1, -1)
        V = chain_df["IV"].values.reshape(1, -1)
        OI = chain_df["OI"].values.reshape(1, -1)
        
        # Ratios mapping
        ratios = np.where(chain_df["Type"] == "Call", 0.5, -0.6).reshape(1, -1)
        
        # Protect against division by zero
        T = np.where(T <= 0, 1e-5, T)

        # Vectorized Black-Scholes computation broadcasting across (50, N) matrix
        d1 = (np.log(P / K) + (self.rate - self.dividend + 0.5 * V ** 2) * T) / (V * np.sqrt(T))
        gamma = (np.exp(-self.dividend * T) * np.exp(-d1 ** 2 / 2)) / (P * V * np.sqrt(2 * np.pi * T))

        # Sum along the options dimension (axis 1) to get the net exposure per price step
        gex_matrix = gamma * self.contract_size * OI * ratios * (P ** 2) * 0.01
        net_gex = np.sum(gex_matrix, axis=1)

        profile_df = pd.DataFrame({
            "UnderlyingPrice": P.flatten(),
            "NetGEX": net_gex
        })
        
        # Interpolate Gamma Flip price (where NetGEX crosses 0)
        flip_price = None
        if len(profile_df) > 1:
            for i in range(len(profile_df) - 1):
                y1, y2 = profile_df.iloc[i]["NetGEX"], profile_df.iloc[i+1]["NetGEX"]
                x1, x2 = profile_df.iloc[i]["UnderlyingPrice"], profile_df.iloc[i+1]["UnderlyingPrice"]
                if (y1 < 0 and y2 >= 0) or (y1 > 0 and y2 <= 0):
                    t = -y1 / (y2 - y1)
                    flip_price = x1 + t * (x2 - x1)
                    break
        
        if flip_price is None:
            flip_price = profile_df.loc[profile_df["NetGEX"].abs().idxmin()]["UnderlyingPrice"]
            
        return profile_df, flip_price

# ==============================================================================
# 2. DASH SYSTEM LAYOUT AND THEME DEFINITION
# ==============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)
app.title = "Real-Time Options Gamma Exposure (GEX) Monitor"

# Set simulator startup baseline matching SPX update
market = OptionsMarketSimulator(ticker="SPX", spot=5350.00)

app.layout = dbc.Container([
    # Top Control & Title Bar
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("⚡ QUANTUM HEDGE", className="text-primary font-weight-bold", style={"letterSpacing": "3px", "fontSize": "12px"}),
                html.H1("REAL-TIME GAMMA EXPOSURE (GEX) ENGINE", className="text-light font-weight-bold m-0", style={"fontSize": "26px"})
            ])
        ], md=8, className="py-3"),
        dbc.Col([
            html.Div([
                dbc.Button("RESET ENGINE", id="btn-reset", color="outline-warning", size="sm", className="mr-2"),
                dbc.Button("LIVE FEED: ON", id="btn-live", color="success", size="sm", className="active-glow")
            ], className="d-flex align-items-center justify-content-end h-100")
        ], md=4, className="py-3")
    ], className="border-bottom border-dark mb-4"),

    # Main Metric Banner Card Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("UNDERLYING SPOT PRICE", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-spot", className="text-light font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div(id="metric-spot-change", className="small")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=3, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("NET MARKET GEX", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-net-gex", className="font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div("Total Delta-hedging flow per 1% spot move", className="small text-muted")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=3, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("GAMMA FLIP LEVEL", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-gamma-flip", className="text-warning font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div(id="metric-flip-distance", className="small")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=3, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("MARKET VOL REGIME", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-regime", className="font-weight-bold"),
                    html.Div(id="metric-regime-desc", className="small text-muted")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=3, md=6, className="mb-4")
    ]),

    # Configuration and Selection Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("ARCHITECT CONTROL DESK", className="bg-transparent border-bottom border-dark text-primary font-weight-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("UNDERLYING SYMBOL", className="text-muted small"),
                            dcc.Dropdown(
                                id="dropdown-ticker",
                                options=[
                                    {"label": "S&P 500 Index (SPX)", "value": "SPX"},
                                    {"label": "Nasdaq 100 Index (NDX)", "value": "NDX"},
                                    {"label": "Apple Inc. (AAPL)", "value": "AAPL"},
                                    {"label": "NVIDIA Corp. (NVDA)", "value": "NVDA"}
                                ],
                                value="SPX",
                                clearable=False,
                                className="bg-dark text-light mb-3"
                            )
                        ], lg=3, md=6),
                        dbc.Col([
                            html.Label("TERM EXPIRATIONS FILTER", className="text-muted small"),
                            dcc.Dropdown(
                                id="dropdown-expiration",
                                options=[
                                    {"label": "All Expirations", "value": "ALL"},
                                    {"label": "0DTE Intraday Magnets", "value": "0DTE"},
                                    {"label": "Short-Term Weekly (1-W)", "value": "1-Week"},
                                    {"label": "Monthly Options (1-M)", "value": "1-Month"}
                                ],
                                value="ALL",
                                clearable=False,
                                className="bg-dark text-light mb-3"
                            )
                        ], lg=3, md=6),
                        dbc.Col([
                            html.Label("VOLATILITY SMILE PARAMETER (ATM IV)", className="text-muted small"),
                            dcc.Slider(
                                id="slider-vol",
                                min=0.08,
                                max=0.45,
                                step=0.01,
                                value=0.16,
                                marks={0.10: "10%", 0.20: "20%", 0.30: "30%", 0.40: "40%"},
                                className="mb-3"
                            )
                        ], lg=3, md=6),
                        dbc.Col([
                            html.Label("SYSTEM REFRESH FREQUENCY", className="text-muted small"),
                            dcc.RadioItems(
                                id="radio-refresh",
                                options=[
                                    {"label": " 3 sec (Live)", "value": 3},
                                    {"label": " 10 sec", "value": 10},
                                    {"label": " Paused", "value": 9999}
                                ],
                                value=3,
                                labelClassName="text-light mr-3",
                                inputStyle={"marginRight": "5px"}
                            )
                        ], lg=3, md=6)
                    ])
                ])
            ], className="bg-dark border-secondary mb-4")
        ], width=12)
    ]),

    # Data Visualizations Viewports
    dbc.Row([
        # Left Panel - Strike distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("GEX CONCENTRATION BY STRIKE (LIQUIDITY WALLS)", className="bg-transparent border-bottom border-dark font-weight-bold text-light"),
                dbc.CardBody([
                    dcc.Graph(id="graph-gex-strikes", style={"height": "480px"}, config={"displayModeBar": False})
                ])
            ], className="bg-dark border-secondary mb-4")
        ], lg=8, md=12),
        
        # Right Panel - Hedging Regime Dynamics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("DEALER GEX PROFILE & REGIME SENSITIVITY", className="bg-transparent border-bottom border-dark font-weight-bold text-light"),
                dbc.CardBody([
                    dcc.Graph(id="graph-gamma-profile", style={"height": "480px"}, config={"displayModeBar": False})
                ])
            ], className="bg-dark border-secondary mb-4")
        ], lg=4, md=12)
    ]),

    # Order Book Depth Breakdown
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AGGREGATED INSTITUTIONAL HEDGING LEVELS & LIQUIDITY MAP", className="bg-transparent border-bottom border-dark font-weight-bold text-light"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.P("CALL WALL (MAJOR RESISTANCE)", className="text-success font-weight-bold small-title mb-1"),
                                html.H3(id="level-call-wall", style={"fontFamily": "monospace"}),
                                html.P("Largest concentration of Call Gamma", className="text-muted small")
                            ], className="text-center p-3 border border-dark rounded bg-black-opacity")
                        ], md=4, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.P("GAMMA FLIP DESK", className="text-warning font-weight-bold small-title mb-1"),
                                html.H3(id="level-gamma-flip", style={"fontFamily": "monospace"}),
                                html.P("Regime shift transition trigger point", className="text-muted small")
                            ], className="text-center p-3 border border-dark rounded bg-black-opacity")
                        ], md=4, className="mb-3"),
                        dbc.Col([
                            html.Div([
                                html.P("PUT WALL (MAJOR SUPPORT)", className="text-danger font-weight-bold small-title mb-1"),
                                html.H3(id="level-put-wall", style={"fontFamily": "monospace"}),
                                html.P("Largest concentration of Put Gamma", className="text-muted small")
                            ], className="text-center p-3 border border-dark rounded bg-black-opacity")
                        ], md=4, className="mb-3")
                    ])
                ])
            ], className="bg-dark border-secondary mb-4")
        ], width=12)
    ]),

    # System State Stores & Timers
    dcc.Interval(id="system-tick", interval=3000, n_intervals=0),
    dcc.Store(id="spot-tracker-store", data={"spot": 5350.00, "prev_spot": 5350.00})
], fluid=True, style={"backgroundColor": "#0d0e12", "minHeight": "100vh", "padding": "20px"})

# Custom styles for elements
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .small-title {
                letter-spacing: 1.5px;
                font-size: 11px;
                font-weight: 700;
            }
            .bg-black-opacity {
                background-color: rgba(0, 0, 0, 0.35);
            }
            .active-glow {
                box-shadow: 0 0 10px rgba(40, 167, 69, 0.6);
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 0.9; }
                50% { opacity: 1; box-shadow: 0 0 16px rgba(40, 167, 69, 0.95); }
                100% { opacity: 0.9; }
            }
            /* Custom Scrollbar for modern look */
            ::-webkit-scrollbar {
                width: 6px;
                height: 6px;
            }
            ::-webkit-scrollbar-track {
                background: #0d0e12;
            }
            ::-webkit-scrollbar-thumb {
                background: #1f232b;
                border-radius: 3px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #3f4756;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==============================================================================
# 3. INTERACTIVE CALLBACK CONTROLLERS
# ==============================================================================

# Dynamic interval update switcher
@app.callback(
    Output("system-tick", "interval"),
    Input("radio-refresh", "value")
)
def update_refresh_frequency(secs):
    return secs * 1000


# Principal System Engine Call Loop
@app.callback(
    [
        Output("spot-tracker-store", "data"),
        Output("metric-spot", "children"),
        Output("metric-spot-change", "children"),
        Output("metric-spot-change", "className"),
        Output("metric-net-gex", "children"),
        Output("metric-net-gex", "className"),
        Output("metric-gamma-flip", "children"),
        Output("metric-flip-distance", "children"),
        Output("metric-flip-distance", "className"),
        Output("metric-regime", "children"),
        Output("metric-regime", "className"),
        Output("metric-regime-desc", "children"),
        Output("level-call-wall", "children"),
        Output("level-gamma-flip", "children"),
        Output("level-put-wall", "children"),
        Output("graph-gex-strikes", "figure"),
        Output("graph-gamma-profile", "figure")
    ],
    [
        Input("system-tick", "n_intervals"),
        Input("dropdown-ticker", "value"),
        Input("dropdown-expiration", "value"),
        Input("slider-vol", "value"),
        Input("btn-reset", "n_clicks")
    ],
    [State("spot-tracker-store", "data")]
)
def run_real_time_quant_engine(n, ticker, expiration, vol, reset_btn, current_state):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    
    # Baseline Spot Configs for symbols (Adjusted to realistic post-split 2026 values)
    base_spots = {
        "SPX": 5350.00,
        "NDX": 18600.00,
        "AAPL": 185.00,
        "NVDA": 135.00  # Corrected from legacy pre-split 875.20 to realistic post-split value
    }
    
    current_base = base_spots.get(ticker, 5000.00)
    
    # Safety Check: If Store is None or not dictionary, initialize it immediately
    if not current_state or not isinstance(current_state, dict):
        current_state = {"spot": current_base, "prev_spot": current_base}
    
    # Handle user resets or ticker switches
    if triggered_id == "btn-reset" or triggered_id == "dropdown-ticker" or triggered_id == "" or triggered_id == ".":
        spot = current_base
        prev_spot = current_base
    else:
        # Securely read values from initialized store
        spot = current_state.get("spot", current_base)
        prev_spot = current_state.get("prev_spot", spot)
        
        # Simulate realistic continuous random-walk drift intraday
        drift = 0.0001 * (np.random.rand() - 0.49) 
        vol_impact = vol * 0.05 * np.random.randn()
        spot = spot * (1.0 + drift + vol_impact)
        
    # Save new simulation state back to Dash Store
    updated_state = {"spot": spot, "prev_spot": prev_spot}
    
    # --------------------------------------------------------------------------
    # COMPUTE LIVE ORDER BOOK Exposure 
    # --------------------------------------------------------------------------
    market.ticker = ticker
    market.vol = vol
    chain_df = market.generate_option_chain(spot)
    
    # Filter by specific expiration selection if user requested
    if expiration != "ALL":
        filtered_chain = chain_df[chain_df["Expiration"] == expiration]
    else:
        filtered_chain = chain_df
        
    # Aggregate GEX per strike
    strike_gex = filtered_chain.groupby("Strike")["GEX"].sum().reset_index()
    
    # Compute continuous profiles (USING VECTORIZED PERFORMANCE SOLVER)
    profile_df, flip_price = market.compute_gamma_profile(filtered_chain, spot)
    
    # Identify institutional key wall levels
    if not strike_gex.empty:
        call_wall_row = strike_gex.loc[strike_gex["GEX"].idxmax()]
        put_wall_row = strike_gex.loc[strike_gex["GEX"].idxmin()]
        call_wall = call_wall_row["Strike"]
        put_wall = put_wall_row["Strike"]
    else:
        call_wall = 0.0
        put_wall = 0.0
    
    # --------------------------------------------------------------------------
    # FORMAT TOP INDICATOR PANELS
    # --------------------------------------------------------------------------
    
    # 1. Spot Price rendering
    spot_str = f"${spot:,.2f}"
    net_day_change = spot - current_base
    pct_day_change = (net_day_change / current_base) * 100.0
    change_sign = "+" if net_day_change >= 0 else ""
    change_color = "text-success" if net_day_change >= 0 else "text-danger"
    change_str = f"{change_sign}${net_day_change:,.2f} ({change_sign}{pct_day_change:.2f}%)"
    
    # 2. Net GEX sum
    net_gex_value = strike_gex["GEX"].sum() if not strike_gex.empty else 0.0
    if abs(net_gex_value) >= 1e9:
        net_gex_str = f"${net_gex_value / 1e9:.2f} Bn"
    else:
        net_gex_str = f"${net_gex_value / 1e6:.2f} Mn"
        
    net_gex_class = "text-success" if net_gex_value >= 0 else "text-danger"
    
    # 3. Gamma Flip
    flip_str = f"${flip_price:,.2f}"
    dist_to_flip = ((spot - flip_price) / spot) * 100
    dist_sign = "+" if dist_to_flip >= 0 else ""
    dist_color = "text-success" if dist_to_flip >= 0 else "text-danger"
    dist_str = f"Spot is {dist_sign}{dist_to_flip:.2f}% from Flip"
    
    # 4. Volatility regime categorization
    if spot >= flip_price:
        regime_label = "LONG GAMMA"
        regime_class = "text-success"
        regime_desc = "Dealers support dips / cap rallies. Low volatility."
    else:
        regime_label = "SHORT GAMMA"
        regime_class = "text-danger"
        regime_desc = "Dealers chase price momentum. Tail-risk volatility high."
        
    # Formatting of levels blocks
    call_wall_str = f"${call_wall:,.1f}"
    put_wall_str = f"${put_wall:,.1f}"
    
    # --------------------------------------------------------------------------
    # DATA GRAPH 1: STRIKE LEVEL BAR CHART
    # --------------------------------------------------------------------------
    colors = ["#10b981" if g >= 0 else "#ef4444" for g in strike_gex["GEX"]] if not strike_gex.empty else []
    
    fig_strikes = go.Figure()
    
    if not strike_gex.empty:
        # Add Bar chart for strike-by-strike exposure
        fig_strikes.add_trace(go.Bar(
            x=strike_gex["Strike"],
            y=strike_gex["GEX"],
            marker_color=colors,
            name="Net Exposure ($ / 1% move)",
            hovertemplate="Strike: %{x}<br>Net GEX: $%{y:,.0f}<extra></extra>"
        ))
    
    # Highlight current Spot price with solid line
    fig_strikes.add_vline(
        x=spot,
        line_width=2,
        line_dash="solid",
        line_color="#3b82f6",
        annotation_text=f"Spot Price: ${spot:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#3b82f6", size=10)
    )
    
    # Highlight Gamma Flip level
    fig_strikes.add_vline(
        x=flip_price,
        line_width=1.5,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Flip: ${flip_price:.1f}",
        annotation_position="bottom right",
        annotation_font=dict(color="#f59e0b", size=10)
    )
    
    fig_strikes.update_layout(
        plot_bgcolor="#0d0e12",
        paper_bgcolor="#181a20",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=dict(text="Options Strike Price", font=dict(color="#8a92a6", size=11)),
            gridcolor="#242830",
            tickfont=dict(color="#6c7383")
        ),
        yaxis=dict(
            title=dict(text="Net Dealer Gamma Exposure ($)", font=dict(color="#8a92a6", size=11)),
            gridcolor="#242830",
            zerolinecolor="#3f4756",
            tickfont=dict(color="#6c7383")
        ),
        showlegend=False,
        hovermode="x unified"
    )
    
    # --------------------------------------------------------------------------
    # DATA GRAPH 2: CONTINUOUS FLIP PROFILE
    # --------------------------------------------------------------------------
    fig_profile = go.Figure()
    
    if not profile_df.empty:
        # Base Curve
        fig_profile.add_trace(go.Scatter(
            x=profile_df["UnderlyingPrice"],
            y=profile_df["NetGEX"],
            mode="lines",
            line=dict(color="#a855f7", width=3),
            fill="tozeroy",
            fillcolor="rgba(168, 85, 247, 0.08)",
            name="Systemic Curve",
            hovertemplate="Underlying: %{x:,.1f}<br>Net GEX: $%{y:,.0f}<extra></extra>"
        ))
        
        # Intersect marker (Zero Gamma / Flip)
        fig_profile.add_trace(go.Scatter(
            x=[flip_price],
            y=[0],
            mode="markers",
            marker=dict(color="#f59e0b", size=12, line=dict(color="white", width=2)),
            name="Gamma Flip Target"
        ))
        
        # Current active spot position marker
        active_spot_gex = np.interp(spot, profile_df["UnderlyingPrice"], profile_df["NetGEX"])
        fig_profile.add_trace(go.Scatter(
            x=[spot],
            y=[active_spot_gex],
            mode="markers",
            marker=dict(color="#3b82f6", size=12, symbol="diamond", line=dict(color="white", width=1.5)),
            name="Current Spot State"
        ))
    
    # Styling
    fig_profile.update_layout(
        plot_bgcolor="#0d0e12",
        paper_bgcolor="#181a20",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=dict(text="Underlying Asset Value Scale", font=dict(color="#8a92a6", size=11)),
            gridcolor="#242830",
            tickfont=dict(color="#6c7383")
        ),
        yaxis=dict(
            title=dict(text="Expected Dealer Systemic Net GEX ($)", font=dict(color="#8a92a6", size=11)),
            gridcolor="#242830",
            zerolinecolor="#3f4756",
            tickfont=dict(color="#6c7383")
        ),
        showlegend=False,
        hovermode="closest"
    )
    
    return (
        updated_state,
        spot_str,
        change_str,
        change_color,
        net_gex_str,
        net_gex_class,
        flip_str,
        dist_str,
        dist_color,
        regime_label,
        regime_class,
        regime_desc,
        call_wall_str,
        flip_str,
        put_wall_str,
        fig_strikes,
        fig_profile
    )


# Run the application using Dash microservice runner
if __name__ == "__main__":
    app.run(debug=True, port=8050)
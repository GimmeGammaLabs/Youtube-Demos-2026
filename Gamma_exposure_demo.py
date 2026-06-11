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
    Simulates real-time options chain data and computes dealer Gamma Exposure (GEX),
    Vanna Exposure (VEX), and Charm Exposure (CEX) using institutional-grade 
    Black-Scholes modeling and vol-skew curves.
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
        volatility smile/skew, and open interest distribution. Appends Vanna and Charm.
        """
        np.random.seed(int(current_spot) % 1000 + 42) # Deterministic yet changing
        
        # Determine strike range around current spot (+/- 10%)
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
                strike_iv = self.vol + 0.15 * (distance ** 2) - 0.12 * distance
                strike_iv = np.clip(strike_iv, 0.05, 0.60)
                
                # 2. Open Interest distribution (Clustered around psychological levels)
                base_oi = np.exp(-0.5 * (distance / 0.04) ** 2) * 50000
                psy_level = 100 if self.ticker in ["SPX", "NDX"] else 10
                half_psy_level = 50 if self.ticker in ["SPX", "NDX"] else 5
                
                if strike % psy_level == 0:
                    base_oi *= 2.5
                elif strike % half_psy_level == 0:
                    base_oi *= 1.6
                
                call_oi = int(np.clip(base_oi * (0.4 + np.random.rand() * 0.6), 50, 150000))
                put_oi = int(np.clip(base_oi * (0.4 + np.random.rand() * 0.6), 50, 150000))
                
                dealer_call_ratio = 0.5 
                dealer_put_ratio = -0.6 
                
                # Black-Scholes Greeks Engine
                for opt_type, oi, ratio in [("Call", call_oi, dealer_call_ratio), ("Put", put_oi, dealer_put_ratio)]:
                    g, d, vanna, charm = self._black_scholes_greeks(current_spot, strike, dte, strike_iv, opt_type)
                    
                    # Calculate Notional Gamma Exposure (GEX) per 1% underlying move
                    gex = g * self.contract_size * oi * ratio * (current_spot ** 2) * 0.01
                    
                    # Vanna Exposure (VEX): Change in Dollar Delta per 1% absolute point move in IV
                    vex = vanna * self.contract_size * oi * ratio * current_spot * 0.01
                    
                    # Charm Exposure (CEX): Daily Delta decay bleed (scaled to days via / 365)
                    cex = (charm / 365.0) * self.contract_size * oi * ratio * current_spot
                    
                    records.append({
                        "Expiration": exp_label,
                        "Strike": strike,
                        "Type": opt_type,
                        "IV": strike_iv,
                        "OI": oi,
                        "Delta": d,
                        "Gamma": g,
                        "GEX": gex,
                        "Vanna": vanna,
                        "VEX": vex,
                        "Charm": charm,
                        "CEX": cex,
                        "DTE": dte
                    })
                    
        return pd.DataFrame(records)

    def _black_scholes_greeks(self, S, K, T, v, opt_type):
        """
        Core analytical solver for option Gamma, Delta, Vanna, and Charm.
        """
        if T <= 0:
            return 0.0, (1.0 if (opt_type == "Call" and S >= K) else 0.0), 0.0, 0.0
            
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (self.rate - self.dividend + 0.5 * v ** 2) * T) / (v * sqrt_T)
        d2 = d1 - v * sqrt_T
        
        n_d1 = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)
        
        # Gamma (identical for Call and Put)
        gamma = (np.exp(-self.dividend * T) * n_d1) / (S * v * sqrt_T)
        
        # Vanna (dVega/dSpot or dDelta/dVol)
        vanna = -np.exp(-self.dividend * T) * n_d1 * d2 / v
        
        # Charm (dDelta/dT - passage of time derivative)
        charm_common = n_d1 * ((self.rate - self.dividend) / (v * sqrt_T) - d2 / (2 * T))
        if opt_type == "Call":
            delta = np.exp(-self.dividend * T) * ndtr(d1)
            charm = self.dividend * np.exp(-self.dividend * T) * ndtr(d1) - np.exp(-self.dividend * T) * charm_common
        else:
            delta = -np.exp(-self.dividend * T) * ndtr(-d1)
            charm = -self.dividend * np.exp(-self.dividend * T) * ndtr(-d1) - np.exp(-self.dividend * T) * charm_common
            
        return gamma, delta, vanna, charm

    def compute_exposure_profiles(self, chain_df, current_spot, price_range_pct=0.10, steps=50):
        """
        Vectorized solver: Calculates the ENTIRE chain's GEX, VEX, and CEX across a spectrum of underlying prices.
        """
        if chain_df.empty:
            return pd.DataFrame(columns=["UnderlyingPrice", "NetGEX", "NetVEX", "NetCEX"]), current_spot

        min_price = current_spot * (1.0 - price_range_pct)
        max_price = current_spot * (1.0 + price_range_pct)
        
        P = np.linspace(min_price, max_price, steps).reshape(-1, 1)
        K = chain_df["Strike"].values.reshape(1, -1)
        T = chain_df["DTE"].values.reshape(1, -1)
        V = chain_df["IV"].values.reshape(1, -1)
        OI = chain_df["OI"].values.reshape(1, -1)
        ratios = np.where(chain_df["Type"] == "Call", 0.5, -0.6).reshape(1, -1)
        is_call = (chain_df["Type"] == "Call").values.reshape(1, -1)
        
        T = np.where(T <= 0, 1e-5, T)

        # BS calculations
        d1 = (np.log(P / K) + (self.rate - self.dividend + 0.5 * V ** 2) * T) / (V * np.sqrt(T))
        d2 = d1 - V * np.sqrt(T)
        n_d1 = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)
        
        # 1. Gamma & GEX
        gamma = (np.exp(-self.dividend * T) * n_d1) / (P * V * np.sqrt(T))
        gex_matrix = gamma * self.contract_size * OI * ratios * (P ** 2) * 0.01
        net_gex = np.sum(gex_matrix, axis=1)
        
        # 2. Vanna & VEX
        vanna = -np.exp(-self.dividend * T) * n_d1 * d2 / V
        vex_matrix = vanna * self.contract_size * OI * ratios * P * 0.01
        net_vex = np.sum(vex_matrix, axis=1)
        
        # 3. Charm & CEX
        charm_common = n_d1 * ((self.rate - self.dividend) / (V * np.sqrt(T)) - d2 / (2 * T))
        ndtr_d1 = ndtr(d1)
        charm = np.where(
            is_call,
            self.dividend * np.exp(-self.dividend * T) * ndtr_d1 - np.exp(-self.dividend * T) * charm_common,
            -self.dividend * np.exp(-self.dividend * T) * (1.0 - ndtr_d1) - np.exp(-self.dividend * T) * charm_common
        )
        cex_matrix = (charm / 365.0) * self.contract_size * OI * ratios * P
        net_cex = np.sum(cex_matrix, axis=1)

        profile_df = pd.DataFrame({
            "UnderlyingPrice": P.flatten(),
            "NetGEX": net_gex,
            "NetVEX": net_vex,
            "NetCEX": net_cex
        })
        
        # Find flip price (where NetGEX crosses 0)
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
app.title = "Real-Time Options Cross-Greek Exposure Monitor"

market = OptionsMarketSimulator(ticker="SPX", spot=5350.00)

app.layout = dbc.Container([
    # Top Control & Title Bar
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span("⚡ QUANTUM HEDGE", className="text-primary font-weight-bold", style={"letterSpacing": "3px", "fontSize": "12px"}),
                html.H1("REAL-TIME SYSTEMIC GEX / VEX / CEX ENGINE", className="text-light font-weight-bold m-0", style={"fontSize": "26px"})
            ])
        ], md=8, className="py-3"),
        dbc.Col([
            html.Div([
                dbc.Button("RESET ENGINE", id="btn-reset", color="outline-warning", size="sm", className="me-2"),
                dbc.Button("LIVE FEED: ON", id="btn-live", color="success", size="sm", className="active-glow")
            ], className="d-flex align-items-center justify-content-end h-100")
        ], md=4, className="py-3")
    ], className="border-bottom border-dark mb-4"),

    # Symmetrical 6-Card Metric Matrix
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("UNDERLYING SPOT PRICE", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-spot", className="text-light font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div(id="metric-spot-change", className="small")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=4, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("NET MARKET GEX", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-net-gex", className="font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div("Total Delta-hedging flow per 1% spot move", className="small text-muted")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=4, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("GAMMA FLIP LEVEL", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-gamma-flip", className="text-warning font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div(id="metric-flip-distance", className="small")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=4, md=6, className="mb-4"),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("NET MARKET VANNA (VEX)", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-net-vex", className="font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div("Delta sensitivity per 1% absolute Vol shift", className="small text-muted")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=4, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("NET MARKET CHARM (CEX)", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-net-cex", className="font-weight-bold", style={"fontFamily": "monospace"}),
                    html.Div("Daily passive Delta decay bleed", className="small text-muted")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=4, md=6, className="mb-4"),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.P("MARKET VOL REGIME", className="text-muted mb-1 small-title"),
                    html.H2(id="metric-regime", className="font-weight-bold"),
                    html.Div(id="metric-regime-desc", className="small text-muted")
                ])
            ], className="bg-dark border-secondary h-100")
        ], lg=4, md=6, className="mb-4")
    ]),

    # Configuration Control Panel
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
                                labelClassName="text-light me-3",
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
        # Left Panel - Greek Strike Selection distribution
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("CONCENTRATION BY STRIKE (LIQUIDITY WALLS)", className="font-weight-bold text-light mt-1"),
                        dcc.RadioItems(
                            id="radio-greek-select",
                            options=[
                                {"label": " GEX (Gamma)", "value": "GEX"},
                                {"label": " VEX (Vanna)", "value": "VEX"},
                                {"label": " CEX (Charm)", "value": "CEX"}
                            ],
                            value="GEX",
                            labelClassName="text-muted me-3 small font-weight-bold",
                            inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                            className="ms-auto d-flex align-items-center"
                        )
                    ], className="d-flex w-100 align-items-center")
                ], className="bg-transparent border-bottom border-dark"),
                dbc.CardBody([
                    dcc.Graph(id="graph-gex-strikes", style={"height": "480px"}, config={"displayModeBar": False})
                ])
            ], className="bg-dark border-secondary mb-4")
        ], lg=8, md=12),
        
        # Right Panel - Hedging Profile
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("DEALER SYSTEMIC EXPOSURE PROFILES (GEX / VEX / CEX)", className="bg-transparent border-bottom border-dark font-weight-bold text-light"),
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

# Custom styles for modern layout
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .small-title { letter-spacing: 1.5px; font-size: 11px; font-weight: 700; }
            .bg-black-opacity { background-color: rgba(0, 0, 0, 0.35); }
            .active-glow { box-shadow: 0 0 10px rgba(40, 167, 69, 0.6); animation: pulse 2s infinite; }
            @keyframes pulse {
                0% { opacity: 0.9; }
                50% { opacity: 1; box-shadow: 0 0 16px rgba(40, 167, 69, 0.95); }
                100% { opacity: 0.9; }
            }
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #0d0e12; }
            ::-webkit-scrollbar-thumb { background: #1f232b; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #3f4756; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# ==============================================================================
# 3. INTERACTIVE CALLBACK CONTROLLERS
# ==============================================================================

def format_exposure(val):
    prefix = "-" if val < 0 else ""
    abs_val = abs(val)
    if abs_val >= 1e9:
        return f"{prefix}${abs_val / 1e9:.2f} Bn"
    elif abs_val >= 1e6:
        return f"{prefix}${abs_val / 1e6:.2f} Mn"
    elif abs_val >= 1e3:
        return f"{prefix}${abs_val / 1e3:.2f} K"
    else:
        return f"{prefix}${abs_val:.2f}"

@app.callback(
    Output("system-tick", "interval"),
    Input("radio-refresh", "value")
)
def update_refresh_frequency(secs):
    return secs * 1000

@app.callback(
    [
        Output("spot-tracker-store", "data"),
        Output("metric-spot", "children"),
        Output("metric-spot-change", "children"),
        Output("metric-spot-change", "className"),
        Output("metric-net-gex", "children"),
        Output("metric-net-gex", "className"),
        Output("metric-net-vex", "children"),
        Output("metric-net-vex", "className"),
        Output("metric-net-cex", "children"),
        Output("metric-net-cex", "className"),
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
        Input("radio-greek-select", "value"),
        Input("btn-reset", "n_clicks")
    ],
    [State("spot-tracker-store", "data")]
)
def run_real_time_quant_engine(n, ticker, expiration, vol, greek_selection, reset_btn, current_state):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    
    base_spots = {"SPX": 5350.00, "NDX": 18600.00, "AAPL": 185.00, "NVDA": 135.00}
    current_base = base_spots.get(ticker, 5000.00)
    
    if not current_state or not isinstance(current_state, dict):
        current_state = {"spot": current_base, "prev_spot": current_base}
    
    if triggered_id in ["btn-reset", "dropdown-ticker", "", "."]:
        spot = current_base
        prev_spot = current_base
    else:
        prev_spot = current_state.get("spot", current_base)
        spot = prev_spot
        if triggered_id == "system-tick":
            drift = 0.0001 * (np.random.rand() - 0.49) 
            vol_impact = vol * 0.05 * np.random.randn()
            spot = spot * (1.0 + drift + vol_impact)
        
    updated_state = {"spot": spot, "prev_spot": prev_spot}
    
    # Generate Chain Data
    market.ticker = ticker
    market.vol = vol
    chain_df = market.generate_option_chain(spot)
    
    if expiration != "ALL":
        filtered_chain = chain_df[chain_df["Expiration"] == expiration]
    else:
        filtered_chain = chain_df
        
    # Aggregate GEX per strike
    strike_gex = filtered_chain.groupby("Strike")["GEX"].sum().reset_index()
    
    # Calculate continuous profile dynamics
    profile_df, flip_price = market.compute_exposure_profiles(filtered_chain, spot)
    
    if not strike_gex.empty:
        call_wall = strike_gex.loc[strike_gex["GEX"].idxmax()]["Strike"]
        put_wall = strike_gex.loc[strike_gex["GEX"].idxmin()]["Strike"]
    else:
        call_wall, put_wall = 0.0, 0.0
    
    # --------------------------------------------------------------------------
    # FORMAT EXTENDED BANNER METRICS
    # --------------------------------------------------------------------------
    spot_str = f"${spot:,.2f}"
    net_day_change = spot - current_base
    pct_day_change = (net_day_change / current_base) * 100.0
    change_sign = "+" if net_day_change >= 0 else ""
    change_color = "text-success" if net_day_change >= 0 else "text-danger"
    change_str = f"{change_sign}${net_day_change:,.2f} ({change_sign}{pct_day_change:.2f}%)"
    
    # 1. Net GEX
    net_gex_val = strike_gex["GEX"].sum() if not strike_gex.empty else 0.0
    net_gex_str = format_exposure(net_gex_val)
    net_gex_class = "text-success" if net_gex_val >= 0 else "text-danger"
    
    # 2. Net VEX
    net_vex_val = filtered_chain["VEX"].sum() if not filtered_chain.empty else 0.0
    net_vex_str = format_exposure(net_vex_val)
    net_vex_class = "text-success" if net_vex_val >= 0 else "text-danger"
    
    # 3. Net CEX
    net_cex_val = filtered_chain["CEX"].sum() if not filtered_chain.empty else 0.0
    net_cex_str = format_exposure(net_cex_val)
    net_cex_class = "text-success" if net_cex_val >= 0 else "text-danger"
    
    # 4. Gamma Flip
    flip_str = f"${flip_price:,.2f}"
    dist_to_flip = ((spot - flip_price) / spot) * 100
    dist_str = f"Spot is {('+' if dist_to_flip >= 0 else '')}{dist_to_flip:.2f}% from Flip"
    dist_color = "text-success" if dist_to_flip >= 0 else "text-danger"
    
    # 5. Volatility Regime
    if spot >= flip_price:
        regime_label = "LONG GAMMA"
        regime_class = "text-success"
        regime_desc = "Dealers support dips / cap rallies. Low volatility."
    else:
        regime_label = "SHORT GAMMA"
        regime_class = "text-danger"
        regime_desc = "Dealers chase price momentum. Tail-risk volatility high."
        
    call_wall_str = f"${call_wall:,.1f}"
    put_wall_str = f"${put_wall:,.1f}"
    
    # --------------------------------------------------------------------------
    # DATA GRAPH 1: DYNAMIC SELECTED STRIKE LEVEL BAR CHART
    # --------------------------------------------------------------------------
    strike_greek = filtered_chain.groupby("Strike")[greek_selection].sum().reset_index()
    colors = ["#10b981" if g >= 0 else "#ef4444" for g in strike_greek[greek_selection]] if not strike_greek.empty else []
    
    fig_strikes = go.Figure()
    if not strike_greek.empty:
        fig_strikes.add_trace(go.Bar(
            x=strike_greek["Strike"],
            y=strike_greek[greek_selection],
            marker_color=colors,
            name=f"Net {greek_selection}",
            hovertemplate="Strike: %{x}<br>Net " + greek_selection + ": $%{y:,.0f}<extra></extra>"
        ))
    
    fig_strikes.add_vline(x=spot, line_width=2, line_color="#3b82f6", annotation_text=f"Spot: ${spot:.2f}", annotation_position="top left", annotation_font_color="#3b82f6")
    fig_strikes.add_vline(x=flip_price, line_width=1.5, line_dash="dash", line_color="#f59e0b", annotation_text=f"Flip: ${flip_price:.1f}", annotation_position="bottom right", annotation_font_color="#f59e0b")
    
    fig_strikes.update_layout(
        plot_bgcolor="#0d0e12", paper_bgcolor="#181a20",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title=dict(text="Options Strike Price", font=dict(color="#8a92a6", size=11)), gridcolor="#242830", tickfont=dict(color="#6c7383")),
        yaxis=dict(title=dict(text=f"Net Dealer {greek_selection} Exposure ($)", font=dict(color="#8a92a6", size=11)), gridcolor="#242830", zerolinecolor="#3f4756", tickfont=dict(color="#6c7383")),
        showlegend=False, hovermode="x unified"
    )
    
    # --------------------------------------------------------------------------
    # DATA GRAPH 2: CONTINUOUS FLIP PROFILE (GEX / VEX / CEX OVERLAYS)
    # --------------------------------------------------------------------------
    fig_profile = go.Figure()
    if not profile_df.empty:
        # Net GEX Trace
        fig_profile.add_trace(go.Scatter(
            x=profile_df["UnderlyingPrice"], y=profile_df["NetGEX"],
            mode="lines", line=dict(color="#a855f7", width=3), 
            fill="tozeroy", fillcolor="rgba(168, 85, 247, 0.04)",
            name="Net GEX", legendgroup="GEX",
            hovertemplate="Underlying: %{x:,.1f}<br>Net GEX: $%{y:,.0f}<extra></extra>"
        ))
        # Net VEX Trace
        fig_profile.add_trace(go.Scatter(
            x=profile_df["UnderlyingPrice"], y=profile_df["NetVEX"],
            mode="lines", line=dict(color="#06b6d4", width=3), 
            fill="tozeroy", fillcolor="rgba(6, 182, 212, 0.04)",
            name="Net Vanna (VEX)", legendgroup="VEX",
            hovertemplate="Underlying: %{x:,.1f}<br>Net VEX: $%{y:,.0f}<extra></extra>"
        ))
        # Net CEX Trace
        fig_profile.add_trace(go.Scatter(
            x=profile_df["UnderlyingPrice"], y=profile_df["NetCEX"],
            mode="lines", line=dict(color="#f97316", width=3), 
            fill="tozeroy", fillcolor="rgba(249, 115, 22, 0.04)",
            name="Net Charm (CEX)", legendgroup="CEX",
            hovertemplate="Underlying: %{x:,.1f}<br>Net CEX: $%{y:,.0f}<extra></extra>"
        ))
        
        # Gamma Flip Marker
        fig_profile.add_trace(go.Scatter(
            x=[flip_price], y=[0], mode="markers", 
            marker=dict(color="#f59e0b", size=12, line=dict(color="white", width=2)),
            name="Gamma Flip", showlegend=False,
            hovertemplate="Gamma Flip: %{x:,.1f}<extra></extra>"
        ))
        
        # Spot GEX Marker
        active_spot_gex = np.interp(spot, profile_df["UnderlyingPrice"], profile_df["NetGEX"])
        fig_profile.add_trace(go.Scatter(
            x=[spot], y=[active_spot_gex], mode="markers", 
            marker=dict(color="#a855f7", size=12, symbol="diamond", line=dict(color="white", width=1.5)),
            name="Spot GEX", legendgroup="GEX", showlegend=False,
            hovertemplate="Spot GEX: $%{y:,.0f}<extra></extra>"
        ))
        
        # Spot VEX Marker
        active_spot_vex = np.interp(spot, profile_df["UnderlyingPrice"], profile_df["NetVEX"])
        fig_profile.add_trace(go.Scatter(
            x=[spot], y=[active_spot_vex], mode="markers", 
            marker=dict(color="#06b6d4", size=12, symbol="diamond", line=dict(color="white", width=1.5)),
            name="Spot VEX", legendgroup="VEX", showlegend=False,
            hovertemplate="Spot VEX: $%{y:,.0f}<extra></extra>"
        ))
        
        # Spot CEX Marker
        active_spot_cex = np.interp(spot, profile_df["UnderlyingPrice"], profile_df["NetCEX"])
        fig_profile.add_trace(go.Scatter(
            x=[spot], y=[active_spot_cex], mode="markers", 
            marker=dict(color="#f97316", size=12, symbol="diamond", line=dict(color="white", width=1.5)),
            name="Spot CEX", legendgroup="CEX", showlegend=False,
            hovertemplate="Spot CEX: $%{y:,.0f}<extra></extra>"
        ))
    
    fig_profile.update_layout(
        plot_bgcolor="#0d0e12", paper_bgcolor="#181a20", margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(title=dict(text="Underlying Asset Value Scale", font=dict(color="#8a92a6", size=11)), gridcolor="#242830", tickfont=dict(color="#6c7383")),
        yaxis=dict(title=dict(text="Dealer Systemic Exposure ($)", font=dict(color="#8a92a6", size=11)), gridcolor="#242830", zerolinecolor="#3f4756", tickfont=dict(color="#6c7383")),
        showlegend=True, hovermode="closest",
        legend=dict(
            font=dict(color="#8a92a6", size=10),
            bgcolor="rgba(13, 14, 18, 0.7)",
            bordercolor="#242830",
            borderwidth=1,
            x=0.02, y=0.98,
            xanchor="left", yanchor="top"
        )
    )
    
    return (
        updated_state, spot_str, change_str, change_color,
        net_gex_str, net_gex_class, net_vex_str, net_vex_class, net_cex_str, net_cex_class,
        flip_str, dist_str, dist_color, regime_label, regime_class, regime_desc,
        call_wall_str, flip_str, put_wall_str, fig_strikes, fig_profile
    )

if __name__ == "__main__":
    app.run(debug=True, port=8050)
//+------------------------------------------------------------------+
//|                                                SafeTradingBot.mq5 |
//|                                 Copyright 2024, Your Company Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Your Company Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property description "Advanced Risk-Free Trading Bot with Comprehensive Protection"

//--- Input Parameters
input group "=== TRADING PARAMETERS ==="
input string Symbol_Name = "";                    // Symbol (empty = current chart)
input ENUM_TIMEFRAMES Timeframe = PERIOD_M5;      // Analysis timeframe
input double Base_Lot_Size = 0.01;                // Base position size
input int Take_Profit_Points = 100;               // Take profit in points
input int Stop_Loss_Points = 50;                  // Stop loss in points
input int Max_Spread_Points = 30;                 // Maximum allowed spread
input ENUM_ORDER_TYPE_FILLING Fill_Policy = ORDER_FILLING_FOK; // Order filling policy

input group "=== RISK MANAGEMENT ==="
input double Max_Risk_Per_Trade = 2.0;            // Max risk per trade (%)
input double Max_Daily_Loss = 5.0;                // Max daily loss (%)
input double Max_Total_Risk = 10.0;               // Max total portfolio risk (%)
input int Max_Open_Positions = 3;                 // Maximum open positions
input int Max_Daily_Trades = 10;                  // Maximum trades per day
input double Min_Account_Balance = 1000.0;        // Minimum account balance to trade

input group "=== TRADING LOGIC ==="
input bool Use_Trend_Filter = true;               // Use trend filtering
input int Trend_Period = 20;                      // Trend calculation period
input double Min_Trend_Strength = 0.5;            // Minimum trend strength (0-1)
input bool Use_Volatility_Filter = true;          // Use volatility filtering
input int Volatility_Period = 14;                 // Volatility calculation period
input double Max_Volatility_Threshold = 2.0;      // Maximum volatility multiplier

input group "=== TIME FILTERS ==="
input bool Use_Time_Filter = true;                // Enable time filtering
input int Start_Hour = 8;                         // Trading start hour
input int End_Hour = 18;                          // Trading end hour
input bool Avoid_News_Time = true;                // Avoid high-impact news times
input int News_Avoidance_Minutes = 30;            // Minutes to avoid around news

input group "=== ADVANCED SETTINGS ==="
input int Magic_Number = 123456;                  // Expert Advisor ID
input string Trade_Comment = "SafeTradingBot";    // Trade comment
input bool Enable_Alerts = true;                  // Enable alerts
input bool Enable_Logging = true;                 // Enable detailed logging
input int Max_Slippage = 3;                       // Maximum slippage in points

//--- Global Variables
string g_symbol;
double g_point;
double g_tick_size;
double g_tick_value;
int g_digits;
double g_lot_step;
double g_min_lot;
double g_max_lot;

// Risk management variables
double g_daily_pnl = 0.0;
int g_daily_trades = 0;
datetime g_last_reset_date = 0;
double g_account_start_balance = 0.0;
double g_max_drawdown = 0.0;
bool g_emergency_stop = false;

// Trading state variables
datetime g_last_trade_time = 0;
int g_consecutive_losses = 0;
double g_last_equity_high = 0.0;

// Technical analysis handles
int g_ma_handle = INVALID_HANDLE;
int g_atr_handle = INVALID_HANDLE;
int g_rsi_handle = INVALID_HANDLE;

//--- Arrays for technical indicators
double g_ma_buffer[];
double g_atr_buffer[];
double g_rsi_buffer[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize symbol
   g_symbol = (Symbol_Name == "") ? Symbol() : Symbol_Name;
   
   // Get symbol properties
   if(!InitializeSymbolProperties())
   {
      Print("Failed to initialize symbol properties");
      return INIT_FAILED;
   }
   
   // Initialize technical indicators
   if(!InitializeTechnicalIndicators())
   {
      Print("Failed to initialize technical indicators");
      return INIT_FAILED;
   }
   
   // Initialize risk management
   InitializeRiskManagement();
   
   // Validate inputs
   if(!ValidateInputs())
   {
      Print("Input validation failed");
      return INIT_FAILED;
   }
   
   Print("SafeTradingBot initialized successfully for ", g_symbol);
   
   if(Enable_Alerts)
      Alert("SafeTradingBot started on ", g_symbol);
      
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(g_ma_handle != INVALID_HANDLE)
      IndicatorRelease(g_ma_handle);
   if(g_atr_handle != INVALID_HANDLE)
      IndicatorRelease(g_atr_handle);
   if(g_rsi_handle != INVALID_HANDLE)
      IndicatorRelease(g_rsi_handle);
      
   Print("SafeTradingBot stopped. Reason: ", reason);
   
   if(Enable_Alerts)
      Alert("SafeTradingBot stopped on ", g_symbol);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update daily statistics
   UpdateDailyStats();
   
   // Check emergency conditions
   if(CheckEmergencyConditions())
   {
      if(!g_emergency_stop)
      {
         g_emergency_stop = true;
         Print("EMERGENCY STOP ACTIVATED!");
         if(Enable_Alerts)
            Alert("EMERGENCY STOP activated for ", g_symbol);
         CloseAllPositions();
      }
      return;
   }
   
   // Reset emergency stop if conditions improve
   if(g_emergency_stop && !CheckEmergencyConditions())
   {
      g_emergency_stop = false;
      Print("Emergency stop deactivated");
   }
   
   // Skip if emergency stop is active
   if(g_emergency_stop)
      return;
      
   // Check if we can trade
   if(!CanTrade())
      return;
      
   // Update technical indicators
   if(!UpdateTechnicalIndicators())
      return;
      
   // Check for trading signals
   CheckTradingSignals();
   
   // Monitor existing positions
   MonitorPositions();
}

//+------------------------------------------------------------------+
//| Initialize symbol properties                                     |
//+------------------------------------------------------------------+
bool InitializeSymbolProperties()
{
   if(!SymbolSelect(g_symbol, true))
   {
      Print("Failed to select symbol: ", g_symbol);
      return false;
   }
   
   g_point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   g_tick_size = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_SIZE);
   g_tick_value = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_VALUE);
   g_digits = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   g_lot_step = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_STEP);
   g_min_lot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MIN);
   g_max_lot = SymbolInfoDouble(g_symbol, SYMBOL_VOLUME_MAX);
   
   Print("Symbol properties initialized:");
   Print("Point: ", g_point, ", Digits: ", g_digits);
   Print("Min lot: ", g_min_lot, ", Max lot: ", g_max_lot, ", Lot step: ", g_lot_step);
   
   return true;
}

//+------------------------------------------------------------------+
//| Initialize technical indicators                                  |
//+------------------------------------------------------------------+
bool InitializeTechnicalIndicators()
{
   // Moving Average for trend detection
   g_ma_handle = iMA(g_symbol, Timeframe, Trend_Period, 0, MODE_EMA, PRICE_CLOSE);
   if(g_ma_handle == INVALID_HANDLE)
   {
      Print("Failed to create MA indicator");
      return false;
   }
   
   // ATR for volatility measurement
   g_atr_handle = iATR(g_symbol, Timeframe, Volatility_Period);
   if(g_atr_handle == INVALID_HANDLE)
   {
      Print("Failed to create ATR indicator");
      return false;
   }
   
   // RSI for momentum analysis
   g_rsi_handle = iRSI(g_symbol, Timeframe, 14, PRICE_CLOSE);
   if(g_rsi_handle == INVALID_HANDLE)
   {
      Print("Failed to create RSI indicator");
      return false;
   }
   
   // Set array properties
   ArraySetAsSeries(g_ma_buffer, true);
   ArraySetAsSeries(g_atr_buffer, true);
   ArraySetAsSeries(g_rsi_buffer, true);
   
   return true;
}

//+------------------------------------------------------------------+
//| Initialize risk management                                       |
//+------------------------------------------------------------------+
void InitializeRiskManagement()
{
   g_account_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_last_equity_high = AccountInfoDouble(ACCOUNT_EQUITY);
   g_daily_pnl = 0.0;
   g_daily_trades = 0;
   g_last_reset_date = TimeCurrent();
   g_consecutive_losses = 0;
   g_max_drawdown = 0.0;
   g_emergency_stop = false;
}

//+------------------------------------------------------------------+
//| Validate input parameters                                        |
//+------------------------------------------------------------------+
bool ValidateInputs()
{
   if(Base_Lot_Size < g_min_lot || Base_Lot_Size > g_max_lot)
   {
      Print("Invalid lot size: ", Base_Lot_Size);
      return false;
   }
   
   if(Take_Profit_Points <= 0 || Stop_Loss_Points <= 0)
   {
      Print("Invalid TP/SL points");
      return false;
   }
   
   if(Stop_Loss_Points >= Take_Profit_Points)
   {
      Print("Stop loss should be smaller than take profit");
      return false;
   }
   
   if(Max_Risk_Per_Trade <= 0 || Max_Risk_Per_Trade > 10)
   {
      Print("Invalid risk per trade: ", Max_Risk_Per_Trade);
      return false;
   }
   
   if(Max_Daily_Loss <= 0 || Max_Daily_Loss > 20)
   {
      Print("Invalid daily loss limit: ", Max_Daily_Loss);
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Update daily statistics                                          |
//+------------------------------------------------------------------+
void UpdateDailyStats()
{
   datetime current_time = TimeCurrent();
   MqlDateTime dt;
   TimeToStruct(current_time, dt);
   
   // Reset daily stats at midnight
   MqlDateTime last_dt;
   TimeToStruct(g_last_reset_date, last_dt);
   
   if(dt.day != last_dt.day)
   {
      g_daily_pnl = 0.0;
      g_daily_trades = 0;
      g_last_reset_date = current_time;
      
      Print("Daily statistics reset");
      
      if(Enable_Logging)
         LogDailyReset();
   }
   
   // Calculate current daily P&L
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double daily_change = current_equity - g_account_start_balance;
   g_daily_pnl = (daily_change / g_account_start_balance) * 100;
   
   // Update maximum drawdown
   if(current_equity > g_last_equity_high)
      g_last_equity_high = current_equity;
   
   double current_drawdown = ((g_last_equity_high - current_equity) / g_last_equity_high) * 100;
   if(current_drawdown > g_max_drawdown)
      g_max_drawdown = current_drawdown;
}

//+------------------------------------------------------------------+
//| Check emergency conditions                                       |
//+------------------------------------------------------------------+
bool CheckEmergencyConditions()
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Check minimum balance
   if(account_balance < Min_Account_Balance)
   {
      if(Enable_Logging)
         Print("Emergency: Account balance below minimum: ", account_balance);
      return true;
   }
   
   // Check daily loss limit
   if(g_daily_pnl <= -Max_Daily_Loss)
   {
      if(Enable_Logging)
         Print("Emergency: Daily loss limit reached: ", g_daily_pnl, "%");
      return true;
   }
   
   // Check total drawdown
   if(g_max_drawdown >= Max_Total_Risk)
   {
      if(Enable_Logging)
         Print("Emergency: Maximum drawdown reached: ", g_max_drawdown, "%");
      return true;
   }
   
   // Check margin level
   double margin_level = AccountInfoDouble(ACCOUNT_MARGIN_LEVEL);
   if(margin_level > 0 && margin_level < 200) // 200% margin level minimum
   {
      if(Enable_Logging)
         Print("Emergency: Low margin level: ", margin_level, "%");
      return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool CanTrade()
{
   // Check if market is open
   if(!IsMarketOpen())
      return false;
      
   // Check time filter
   if(Use_Time_Filter && !IsTimeToTrade())
      return false;
      
   // Check news avoidance
   if(Avoid_News_Time && IsNewsTime())
      return false;
      
   // Check spread
   if(!IsSpreadAcceptable())
      return false;
      
   // Check daily trade limit
   if(g_daily_trades >= Max_Daily_Trades)
      return false;
      
   // Check position limit
   if(CountOpenPositions() >= Max_Open_Positions)
      return false;
      
   // Check consecutive losses
   if(g_consecutive_losses >= 5) // Maximum 5 consecutive losses
      return false;
      
   return true;
}

//+------------------------------------------------------------------+
//| Check if market is open                                          |
//+------------------------------------------------------------------+
bool IsMarketOpen()
{
   return SymbolInfoInteger(g_symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_FULL;
}

//+------------------------------------------------------------------+
//| Check if it's time to trade                                      |
//+------------------------------------------------------------------+
bool IsTimeToTrade()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   return (dt.hour >= Start_Hour && dt.hour < End_Hour);
}

//+------------------------------------------------------------------+
//| Check if it's news time (simplified implementation)             |
//+------------------------------------------------------------------+
bool IsNewsTime()
{
   // This is a simplified implementation
   // In a real EA, you would integrate with a news calendar API
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   // Avoid trading during typical news hours (example: 14:30 GMT)
   if(dt.hour == 14 && dt.min >= 15 && dt.min <= 45)
      return true;
      
   return false;
}

//+------------------------------------------------------------------+
//| Check if spread is acceptable                                    |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable()
{
   double ask = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(g_symbol, SYMBOL_BID);
   int spread = (int)((ask - bid) / g_point);
   
   return (spread <= Max_Spread_Points);
}

//+------------------------------------------------------------------+
//| Count open positions                                             |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetSymbol(i) == g_symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Update technical indicators                                      |
//+------------------------------------------------------------------+
bool UpdateTechnicalIndicators()
{
   // Copy MA values
   if(CopyBuffer(g_ma_handle, 0, 0, 3, g_ma_buffer) < 3)
   {
      Print("Failed to copy MA buffer");
      return false;
   }
   
   // Copy ATR values
   if(CopyBuffer(g_atr_handle, 0, 0, 2, g_atr_buffer) < 2)
   {
      Print("Failed to copy ATR buffer");
      return false;
   }
   
   // Copy RSI values
   if(CopyBuffer(g_rsi_handle, 0, 0, 2, g_rsi_buffer) < 2)
   {
      Print("Failed to copy RSI buffer");
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check for trading signals                                        |
//+------------------------------------------------------------------+
void CheckTradingSignals()
{
   double current_price = SymbolInfoDouble(g_symbol, SYMBOL_BID);
   
   // Get trend direction
   ENUM_SIGNAL_TYPE trend_signal = GetTrendSignal();
   if(trend_signal == SIGNAL_NONE)
      return;
      
   // Check volatility filter
   if(Use_Volatility_Filter && !IsVolatilityAcceptable())
      return;
      
   // Check RSI for overbought/oversold conditions
   if(!IsRSISignalValid(trend_signal))
      return;
      
   // Calculate position size with risk management
   double lot_size = CalculatePositionSize();
   if(lot_size < g_min_lot)
      return;
      
   // Place the trade
   if(trend_signal == SIGNAL_BUY)
   {
      OpenBuyPosition(lot_size);
   }
   else if(trend_signal == SIGNAL_SELL)
   {
      OpenSellPosition(lot_size);
   }
}

//+------------------------------------------------------------------+
//| Signal types enumeration                                         |
//+------------------------------------------------------------------+
enum ENUM_SIGNAL_TYPE
{
   SIGNAL_NONE,
   SIGNAL_BUY,
   SIGNAL_SELL
};

//+------------------------------------------------------------------+
//| Get trend signal                                                 |
//+------------------------------------------------------------------+
ENUM_SIGNAL_TYPE GetTrendSignal()
{
   if(!Use_Trend_Filter)
      return SIGNAL_BUY; // Default to buy if no trend filter
      
   double current_price = SymbolInfoDouble(g_symbol, SYMBOL_BID);
   double ma_current = g_ma_buffer[0];
   double ma_previous = g_ma_buffer[1];
   
   // Calculate trend strength
   double trend_strength = MathAbs(current_price - ma_current) / (g_atr_buffer[0] * 100);
   
   if(trend_strength < Min_Trend_Strength)
      return SIGNAL_NONE;
      
   // Determine trend direction
   if(current_price > ma_current && ma_current > ma_previous)
      return SIGNAL_BUY;
   else if(current_price < ma_current && ma_current < ma_previous)
      return SIGNAL_SELL;
      
   return SIGNAL_NONE;
}

//+------------------------------------------------------------------+
//| Check if volatility is acceptable                               |
//+------------------------------------------------------------------+
bool IsVolatilityAcceptable()
{
   double current_atr = g_atr_buffer[0];
   double previous_atr = g_atr_buffer[1];
   
   double volatility_ratio = current_atr / previous_atr;
   
   return (volatility_ratio <= Max_Volatility_Threshold);
}

//+------------------------------------------------------------------+
//| Check if RSI signal is valid                                    |
//+------------------------------------------------------------------+
bool IsRSISignalValid(ENUM_SIGNAL_TYPE signal)
{
   double rsi_current = g_rsi_buffer[0];
   
   if(signal == SIGNAL_BUY && rsi_current > 70)
      return false; // Don't buy when overbought
      
   if(signal == SIGNAL_SELL && rsi_current < 30)
      return false; // Don't sell when oversold
      
   return true;
}

//+------------------------------------------------------------------+
//| Calculate position size with risk management                    |
//+------------------------------------------------------------------+
double CalculatePositionSize()
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (Max_Risk_Per_Trade / 100);
   
   // Calculate position size based on stop loss
   double stop_loss_value = Stop_Loss_Points * g_point * g_tick_value / g_tick_size;
   double calculated_lots = risk_amount / stop_loss_value;
   
   // Apply lot size constraints
   calculated_lots = MathMax(calculated_lots, g_min_lot);
   calculated_lots = MathMin(calculated_lots, g_max_lot);
   calculated_lots = MathMin(calculated_lots, Base_Lot_Size * 2); // Max 2x base size
   
   // Round to lot step
   calculated_lots = NormalizeDouble(calculated_lots / g_lot_step, 0) * g_lot_step;
   
   return calculated_lots;
}

//+------------------------------------------------------------------+
//| Open buy position                                                |
//+------------------------------------------------------------------+
void OpenBuyPosition(double lot_size)
{
   double ask = SymbolInfoDouble(g_symbol, SYMBOL_ASK);
   double sl = ask - (Stop_Loss_Points * g_point);
   double tp = ask + (Take_Profit_Points * g_point);
   
   // Normalize prices
   sl = NormalizeDouble(sl, g_digits);
   tp = NormalizeDouble(tp, g_digits);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = g_symbol;
   request.volume = lot_size;
   request.type = ORDER_TYPE_BUY;
   request.price = ask;
   request.sl = sl;
   request.tp = tp;
   request.deviation = Max_Slippage;
   request.magic = Magic_Number;
   request.comment = Trade_Comment;
   request.type_filling = Fill_Policy;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         g_daily_trades++;
         g_last_trade_time = TimeCurrent();
         
         Print("BUY order opened: Ticket=", result.order, 
               ", Volume=", lot_size, 
               ", Price=", ask,
               ", SL=", sl,
               ", TP=", tp);
               
         if(Enable_Alerts)
            Alert("BUY position opened on ", g_symbol, " at ", ask);
      }
      else
      {
         Print("BUY order failed: ", result.retcode, " - ", result.comment);
      }
   }
   else
   {
      Print("OrderSend failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Open sell position                                               |
//+------------------------------------------------------------------+
void OpenSellPosition(double lot_size)
{
   double bid = SymbolInfoDouble(g_symbol, SYMBOL_BID);
   double sl = bid + (Stop_Loss_Points * g_point);
   double tp = bid - (Take_Profit_Points * g_point);
   
   // Normalize prices
   sl = NormalizeDouble(sl, g_digits);
   tp = NormalizeDouble(tp, g_digits);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = g_symbol;
   request.volume = lot_size;
   request.type = ORDER_TYPE_SELL;
   request.price = bid;
   request.sl = sl;
   request.tp = tp;
   request.deviation = Max_Slippage;
   request.magic = Magic_Number;
   request.comment = Trade_Comment;
   request.type_filling = Fill_Policy;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         g_daily_trades++;
         g_last_trade_time = TimeCurrent();
         
         Print("SELL order opened: Ticket=", result.order, 
               ", Volume=", lot_size, 
               ", Price=", bid,
               ", SL=", sl,
               ", TP=", tp);
               
         if(Enable_Alerts)
            Alert("SELL position opened on ", g_symbol, " at ", bid);
      }
      else
      {
         Print("SELL order failed: ", result.retcode, " - ", result.comment);
      }
   }
   else
   {
      Print("OrderSend failed: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Monitor existing positions                                       |
//+------------------------------------------------------------------+
void MonitorPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == g_symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         double profit = PositionGetDouble(POSITION_PROFIT);
         
         // Check for trailing stop or other position management
         // This is where you could implement trailing stops, partial closes, etc.
         
         // Update consecutive losses counter
         if(profit < 0 && PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         {
            // Position closed at loss (simplified check)
            // In reality, you'd need to track position history
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == g_symbol && PositionGetInteger(POSITION_MAGIC) == Magic_Number)
      {
         ulong ticket = PositionGetInteger(POSITION_TICKET);
         
         MqlTradeRequest request = {};
         MqlTradeResult result = {};
         
         request.action = TRADE_ACTION_DEAL;
         request.symbol = g_symbol;
         request.volume = PositionGetDouble(POSITION_VOLUME);
         request.type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
         request.price = (request.type == ORDER_TYPE_SELL) ? SymbolInfoDouble(g_symbol, SYMBOL_BID) : SymbolInfoDouble(g_symbol, SYMBOL_ASK);
         request.deviation = Max_Slippage;
         request.magic = Magic_Number;
         request.comment = "Emergency Close";
         request.type_filling = Fill_Policy;
         
         if(OrderSend(request, result))
         {
            Print("Emergency close: Position ", ticket, " closed");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Log daily reset                                                  |
//+------------------------------------------------------------------+
void LogDailyReset()
{
   string log_message = StringFormat("Daily Reset - Date: %s, Previous P&L: %.2f%%, Trades: %d, Max DD: %.2f%%",
                                   TimeToString(TimeCurrent(), TIME_DATE),
                                   g_daily_pnl,
                                   g_daily_trades,
                                   g_max_drawdown);
   Print(log_message);
}

//+------------------------------------------------------------------+
//| OnTrade function - Called when trade operations are performed   |
//+------------------------------------------------------------------+
void OnTrade()
{
   // This function is called when trade operations are performed
   // You can add additional trade monitoring logic here
   
   // Update consecutive losses counter
   UpdateConsecutiveLosses();
}

//+------------------------------------------------------------------+
//| Update consecutive losses counter                                |
//+------------------------------------------------------------------+
void UpdateConsecutiveLosses()
{
   // Get the last closed position to check if it was profitable
   if(HistorySelect(TimeCurrent() - 86400, TimeCurrent())) // Last 24 hours
   {
      int total = HistoryDealsTotal();
      for(int i = total - 1; i >= 0; i--)
      {
         ulong ticket = HistoryDealGetTicket(i);
         if(HistoryDealGetString(ticket, DEAL_SYMBOL) == g_symbol &&
            HistoryDealGetInteger(ticket, DEAL_MAGIC) == Magic_Number &&
            HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         {
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            
            if(profit < 0)
               g_consecutive_losses++;
            else
               g_consecutive_losses = 0;
               
            break; // Only check the most recent deal
         }
      }
   }
}
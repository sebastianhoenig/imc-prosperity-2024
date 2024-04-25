import collections

import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import json
import numpy as np
from math import log, sqrt, exp, erf

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()


def get_price_info(order_depth: OrderDepth):
    sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
    buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

    # Orders are in form of Dict[price, amount]
    if list(buy_orders.items()) == []:
        return 0, 0, 0, 0, 0
    best_sell_price, _ = list(sell_orders.items())[0]
    best_buy_price, _ = list(buy_orders.items())[0]

    worst_sell_price, _ = list(sell_orders.items())[-1]
    worst_buy_price, _ = list(buy_orders.items())[-1]


    vol = 0
    tot = 0
    for ask_price, volume in sell_orders.items():
        tot += (ask_price * -volume)
        vol -= volume
    for bid_price, volume in buy_orders.items():
        tot += (bid_price * volume)
        vol += volume

    curr_mid_price = int(round(tot / vol))

    return curr_mid_price, best_buy_price, best_sell_price, worst_buy_price, worst_sell_price


def blackScholes(r, S, K, T, sigma):
    """Calculate BS price of call/put"""
    d1 = (log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    price = S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
    return int(round(price)), normal_cdf(d1)


def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1 + erf(x / sqrt(2))) / 2

class Trader:

    def __init__(self):
        self.mav_basket_5 = []
        self.mav_basket_long = []
        self.orchids_bid = None
        self.orchids_ask = None
        self.orchids_transport_fees = None
        self.orchids_export_tariff = None
        self.orchids_import_tariff = None
        self.orchids_sunlight = None
        self.orchids_humidity = None
        self.POSITION_LIMITS = {"AMETHYSTS": 20, "STARFRUIT": 20, "ORCHIDS": 100, "CHOCOLATE": 250, "STRAWBERRIES": 350,
                                "ROSES": 60, "GIFT_BASKET": 60, "COCONUT": 300, "COCONUT_COUPON": 600}
        self.position = {"AMETHYSTS": 0, "STARFRUIT": 0, "ORCHIDS": 0, "CHOCOLATE": 0, "STRAWBERRIES": 0, "ROSES": 0,
                         "GIFT_BASKET": 0, "COCONUT": 0, "COCONUT_COUPON": 0}
        self.starfruit_cache = []
        self.own_trades_orchids = []
        self.cache_limit = 17
        self.timestep = 0
        self.roses_were_at_highest = False
        self.roses_were_at_lowest = False
        self.roses_highest_price = 0
        self.roses_lowest_price = 10000
        self.rhianna_bought_roses = False
        self.rhianna_sold_roses = False

    def calc_next_price_starfruit(self, buy_orders, sell_orders):
        X = np.array([i for i in range(len(self.starfruit_cache))])
        Y = np.array(self.starfruit_cache)
        z = np.polyfit(X, Y, 1)
        p = np.poly1d(z)
        nxt_price = p(len(self.starfruit_cache))
        return int(round(nxt_price))

    def trade_starfruit_lr_last_few_timesteps(self, order_depth: OrderDepth, product: str):
        orders: List[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        position = self.position[product]

        if len(self.starfruit_cache) == self.cache_limit:
            self.starfruit_cache.pop(0)

        curr_mid_price, best_buy_price, best_sell_price, worst_buy_price, worst_sell_price = get_price_info(order_depth)
        self.starfruit_cache.append(curr_mid_price)

        if len(self.starfruit_cache) == self.cache_limit:
            next_price = self.calc_next_price_starfruit(buy_orders, sell_orders)
            bid = next_price - 1
            ask = next_price + 1

            for ask_price, volume in sell_orders.items():
                if ((ask_price <= bid) or (position < 0 and ask_price == bid+1)) and position < self.POSITION_LIMITS[product]:
                    max_amount_to_buy = self.POSITION_LIMITS[product] - position
                    order_for = min(-volume, max_amount_to_buy)
                    position += order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, ask_price, order_for))

            undercut_buy = min(best_buy_price + 1, bid)

            if position < self.POSITION_LIMITS[product]:
                possible_volume = self.POSITION_LIMITS[product] - position
                orders.append(Order(product, undercut_buy, possible_volume))

            position = self.position[product]

            max_amount_to_sell = self.POSITION_LIMITS[product] + position
            for bid_price, volume in buy_orders.items():
                if ((bid_price >= ask) or (position > 5 and bid_price == ask-1)) and position > -self.POSITION_LIMITS[product]:
                    order_for = min(volume, max_amount_to_sell)
                    position -= order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, bid_price, -order_for))

            overcut_sell = max(best_sell_price - 1, ask)

            if position > -self.POSITION_LIMITS[product]:
                possible_volume = -self.POSITION_LIMITS[product] - position
                orders.append(Order(product, overcut_sell, possible_volume))

        return orders

    def trade_amethysts_best_bid_best_ask(self, order_depth: OrderDepth, product: str):
        bid = ask = 10000
        orders: List[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        best_sell_price, best_sell_amount = list(sell_orders.items())[0]
        best_buy_price, best_buy_amount = list(buy_orders.items())[0]

        position = self.position[product]

        for ask_price, volume in sell_orders.items():
            if ((ask_price < bid) or (position < -5 and ask_price == bid)) and position < self.POSITION_LIMITS[product]:
                max_amount_to_buy = self.POSITION_LIMITS[product] - position
                order_for = min(-volume, max_amount_to_buy)
                position += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask_price, order_for))

        undercut_buy = best_buy_price + 1
        bid_pr = min(undercut_buy, bid - 1)

        if position < self.POSITION_LIMITS[product]:
            possible_volume = self.POSITION_LIMITS[product] - position
            orders.append(Order(product, bid_pr, possible_volume))
            position += possible_volume

        position = self.position[product]

        max_amount_to_sell = self.POSITION_LIMITS[product] + position
        for bid_price, volume in buy_orders.items():
            if ((bid_price > ask) or (position > 0 and bid_price == ask)) and position > -self.POSITION_LIMITS[product]:
                order_for = min(volume, max_amount_to_sell)
                position -= order_for
                assert (order_for >= 0)
                orders.append(Order(product, bid_price, -order_for))

        overcut_sell = best_sell_price - 1
        sell_pr = max(overcut_sell, ask + 1)

        if position > -self.POSITION_LIMITS[product]:
            possible_volume = -self.POSITION_LIMITS[product] - position
            orders.append(Order(product, sell_pr, possible_volume))

        return orders

    def update_orchids_information(self, orchids):
        self.orchids_bid = orchids.bidPrice
        self.orchids_ask = orchids.askPrice
        self.orchids_transport_fees = orchids.transportFees
        self.orchids_export_tariff = orchids.exportTariff
        self.orchids_import_tariff = orchids.importTariff
        self.orchids_sunlight = orchids.sunlight
        self.orchids_humidity = orchids.humidity

    def trade_orchids(self, order_depth: OrderDepth, product: str):
        orders: List[Order] = []
        # If you want to purchase 1 unit of ORCHID from the south, you will purchase at the askPrice,
        # pay the TRANSPORT_FEES, IMPORT_TARIFF
        purchase_price_from_south = self.orchids_ask + self.orchids_transport_fees + self.orchids_import_tariff

        # If you want to sell 1 unit of ORCHID to the south, you will sell at the bidPrice, pay the
        # TRANSPORT_FEES, EXPORT_TARIFF

        sell_price_to_south = self.orchids_bid - self.orchids_transport_fees - self.orchids_export_tariff

        our_mid_price, our_best_bid, our_best_ask, _, _ = get_price_info(order_depth)

        position = 0

        diff_buy_from_south = our_mid_price - purchase_price_from_south
        diff_sell_to_south = sell_price_to_south - our_mid_price

        if diff_buy_from_south < diff_sell_to_south:
            for ask_price, volume in order_depth.sell_orders.items():
                if ask_price < sell_price_to_south and position < self.POSITION_LIMITS[product]:
                    max_amount_to_buy = self.POSITION_LIMITS[product] - position
                    order_for = min(-volume, max_amount_to_buy)
                    position += order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, ask_price, order_for))
            if not (self.timestep % 1000000 == 0 and self.timestep != 0):
                if position < self.POSITION_LIMITS[product] and ((our_mid_price + 3) < sell_price_to_south):
                    possible_volume = self.POSITION_LIMITS[product] - position
                    try_buy_at_p_1 = int(4/6 * possible_volume)
                    try_buy_at_p_2 = int(2/6 * possible_volume)
                    try_buy_at_p_3 = possible_volume - try_buy_at_p_1 - try_buy_at_p_2
                    orders.append(Order(product, our_mid_price + 1, try_buy_at_p_1))
                    orders.append(Order(product, our_mid_price + 2, try_buy_at_p_2))
                    orders.append(Order(product, our_mid_price + 3, try_buy_at_p_3))
                    position += possible_volume
                elif position < self.POSITION_LIMITS[product] and ((our_mid_price + 2) < sell_price_to_south):
                    possible_volume = self.POSITION_LIMITS[product] - position
                    try_buy_at_p_1 = int(5/6 * possible_volume)
                    try_buy_at_p_2 = possible_volume - try_buy_at_p_1
                    orders.append(Order(product, our_mid_price + 1, try_buy_at_p_1))
                    orders.append(Order(product, our_mid_price + 2, try_buy_at_p_2))
                    position += possible_volume
                elif position < self.POSITION_LIMITS[product] and ((our_mid_price + 1) < sell_price_to_south):
                    possible_volume = self.POSITION_LIMITS[product] - position
                    orders.append(Order(product, our_mid_price + 1, possible_volume))
                    position += possible_volume
        else:
            for bid_price, volume in order_depth.buy_orders.items():
                if bid_price > purchase_price_from_south and position > -self.POSITION_LIMITS[product]:
                    max_amount_to_sell = self.POSITION_LIMITS[product] + position
                    order_for = min(volume, max_amount_to_sell)
                    position -= order_for
                    assert (order_for >= 0)
                    orders.append(Order(product, bid_price, -order_for))
            if not (self.timestep % 1000000 == 0 and self.timestep != 0):
                if position > -self.POSITION_LIMITS[product] and ((our_mid_price - 3) > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    try_sell_at_p_1 = int(7/12 * possible_volume)
                    try_sell_at_p_2 = int(5/12 * possible_volume)
                    try_sell_at_p_3 = possible_volume - try_sell_at_p_1 - try_sell_at_p_2
                    orders.append(Order(product, our_mid_price - 1, try_sell_at_p_1))
                    orders.append(Order(product, our_mid_price - 2, try_sell_at_p_2))
                    orders.append(Order(product, (our_mid_price - 3), try_sell_at_p_3))
                elif position > -self.POSITION_LIMITS[product] and ((our_mid_price - 2) > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    try_sell_at_p_1 = int(9/12 * possible_volume)
                    try_sell_at_p_2 = possible_volume - try_sell_at_p_1
                    orders.append(Order(product, our_mid_price - 1, try_sell_at_p_1))
                    orders.append(Order(product, our_mid_price - 2, try_sell_at_p_2))
                elif position > -self.POSITION_LIMITS[product] and ((our_mid_price - 1) > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    orders.append(Order(product, our_mid_price - 1, possible_volume))
                elif position > -self.POSITION_LIMITS[product] and (our_mid_price > purchase_price_from_south):
                    possible_volume = -self.POSITION_LIMITS[product] - position
                    orders.append(Order(product, our_mid_price, possible_volume))
        conversions = -self.position[product]

        return orders, conversions

    def trade_options(self, order_depths: dict[str, OrderDepth]) -> dict[str, list[Order]]:
        products = ["COCONUT", "COCONUT_COUPON"]
        orders = {"COCONUT": [], "COCONUT_COUPON": []}

        best_sell, best_buy, worst_sell, worst_buy, mid_price, = {}, {}, {}, {}, {}

        for product in products:
            order_depth: OrderDepth = order_depths[product]

            p_mid_price, p_best_buy, p_best_sell, p_worst_buy, p_worst_sell = get_price_info(order_depth)
            mid_price[product] = p_mid_price
            best_buy[product] = p_best_buy
            best_sell[product] = p_best_sell
            worst_buy[product] = p_worst_buy
            worst_sell[product] = p_worst_sell
        T = 245/252
        r = 0
        K = 10000
        sigma = 0.1609616171503603
        S = mid_price["COCONUT"]

        fair_price_option, delta = blackScholes(r, S, K, T, sigma)

        diff = fair_price_option - mid_price["COCONUT_COUPON"]

        desired_pos = -int(delta * self.position["COCONUT_COUPON"])

        current_pos = self.position["COCONUT"]

        amount_to_handle = abs(current_pos - desired_pos)

        if desired_pos < current_pos:
            orders["COCONUT"].append(Order("COCONUT", best_buy["COCONUT"], -amount_to_handle))
        elif desired_pos > current_pos:
            orders["COCONUT"].append(Order("COCONUT", best_sell["COCONUT"], amount_to_handle))

        fair_bid = int(round(min(fair_price_option - 5, best_buy["COCONUT_COUPON"]+1)))
        fair_ask = int(round(max(fair_price_option + 5, best_sell["COCONUT_COUPON"]-1)))

        position = self.position["COCONUT_COUPON"]

        sell_orders = collections.OrderedDict(sorted(order_depths["COCONUT_COUPON"].sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depths["COCONUT_COUPON"].buy_orders.items(), reverse=True))

        for ask_price, volume in sell_orders.items():
            if (ask_price <= fair_bid) and position < self.POSITION_LIMITS["COCONUT_COUPON"]:
                max_amount_to_buy = self.POSITION_LIMITS["COCONUT_COUPON"] - position
                order_for = min(-volume, max_amount_to_buy)
                position += order_for
                assert (order_for >= 0)
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", ask_price, order_for))

        if diff > 5:
            if position < self.POSITION_LIMITS["COCONUT_COUPON"]:
                possible_volume = self.POSITION_LIMITS["COCONUT_COUPON"] - position
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", fair_bid, possible_volume))
                position += possible_volume

        position = self.position["COCONUT_COUPON"]

        max_amount_to_sell = self.POSITION_LIMITS["COCONUT_COUPON"] + position
        for bid_price, volume in buy_orders.items():
            if (bid_price >= fair_ask) and position > -self.POSITION_LIMITS["COCONUT_COUPON"]:
                order_for = min(volume, max_amount_to_sell)
                position -= order_for
                assert (order_for >= 0)
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", bid_price, -order_for))

        if diff < -5:
            if position > -self.POSITION_LIMITS["COCONUT_COUPON"]:
                possible_volume = -self.POSITION_LIMITS["COCONUT_COUPON"] - position
                orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", fair_ask, possible_volume))

        return orders

    def cache_results(self):
        return {
            "starfruit_cache": self.starfruit_cache,
            "roses_were_at_highest": self.roses_were_at_highest,
            "roses_highest_price": self.roses_highest_price,
            "roses_were_at_lowest": self.roses_were_at_lowest,
            "mav_basket_long": self.mav_basket_long,
            "mav_basket_5": self.mav_basket_5,
            "sf_last_price": self.sf_last_price
        }

    def uncache_results(self, trader_data):
        if trader_data:
            data = jsonpickle.decode(trader_data)
            self.starfruit_cache = data["starfruit_cache"]
            self.roses_were_at_highest = data["roses_were_at_highest"]
            self.roses_were_at_lowest = data["roses_were_at_lowest"]
            self.roses_highest_price = data["roses_highest_price"]
            self.mav_basket_long = data["mav_basket_long"]
            self.mav_basket_5 = data["mav_basket_5"]
            self.sf_last_price = data["sf_last_price"]

    def calculate_orders(self, product, order_depth, our_bid, our_ask):
        orders: list[Order] = []

        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        position = self.position[product]
        limit = self.POSITION_LIMITS[product]

        bid_price = our_bid
        ask_price = our_ask

        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid + 1)):
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        position = self.position[product]

        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid + 1 == our_ask)):
                num_orders = max(-vol, -limit - position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders

        return orders

    def did_the_trader_sell(self, state: TradingState, trader_name: str, product: str):
        market_trades = state.market_trades.get(product, [])
        if market_trades:
            for t in market_trades:
                if t.seller == trader_name:
                    return True
        return False

    def did_the_trader_buy(self, state: TradingState, trader_name: str, product: str):
        market_trades = state.market_trades.get(product, [])
        if market_trades:
            for t in market_trades:
                if t.buyer == trader_name:
                    return True
        return False

    def update_buy_sell_roses(self, state: TradingState):
        self.rhianna_bought_roses = self.did_the_trader_buy(state, "Rhianna", "ROSES")
        self.rhianna_sold_roses = self.did_the_trader_sell(state, "Rhianna", "ROSES")

    def trade_roses_with_rhianna(self, order_depth: OrderDepth, product: str):
        orders: List[Order] = []

        position = self.position[product]

        curr_mid_price, best_buy_price, best_sell_price, worst_buy_price, worst_sell_price = get_price_info(order_depth)

        """if not self.roses_were_at_highest: # too risky
            self.roses_highest_price = max(self.roses_highest_price, curr_mid_price)
            if position < self.POSITION_LIMITS["ROSES"]:
                possible_volume = self.POSITION_LIMITS["ROSES"] - position
                orders.append(Order("ROSES", worst_sell_price, possible_volume))
                position += possible_volume"""

        if self.rhianna_bought_roses:
            if position < self.POSITION_LIMITS["ROSES"]:
                possible_volume = self.POSITION_LIMITS["ROSES"] - position
                orders.append(Order("ROSES", worst_sell_price, possible_volume))
                position += possible_volume

        if self.rhianna_sold_roses:
            self.roses_were_at_highest = True
            if position > -self.POSITION_LIMITS["ROSES"]:
                possible_volume = -self.POSITION_LIMITS["ROSES"] - position
                orders.append(Order("ROSES", worst_buy_price, possible_volume))
                position += possible_volume

        return orders

    def run(self, state: TradingState):
        conversion_observations = state.observations.conversionObservations
        self.own_trades_orchids = state.own_trades.get("ORCHIDS", [])
        result = {}
        conversions = -1
        self.uncache_results(state.traderData)
        self.timestep = state.timestamp
        for product in self.position:
            position: int = state.position.get(product, 0)
            self.position[product] = position
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            if product == "AMETHYSTS":
                orders = self.trade_amethysts_best_bid_best_ask(order_depth, product)
                result[product] = orders
            elif product == "STARFRUIT":
                orders = self.trade_starfruit_lr_last_few_timesteps(order_depth, product)
                result[product] = orders
            elif product == "ORCHIDS":
                self.update_orchids_information(conversion_observations["ORCHIDS"])
                orders, conversions = self.trade_orchids(order_depth, product)
                result[product] = orders
            elif product == 'GIFT_BASKET':
                DIFFERENCE_MEAN = 380
                DIFFERENCE_STD = 76.42438217375009
                PERCENT_OF_STD_TO_TRADE_AT = 0.5
                basket_items = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
                mid_price = {}

                for item in basket_items:
                    _, best_buy_price, best_sell_price, _, _ = get_price_info(state.order_depths[item])
                    mid_price[item] = (best_sell_price + best_buy_price) / 2

                difference = mid_price['GIFT_BASKET'] - 4 * mid_price['CHOCOLATE'] - 6 * mid_price['STRAWBERRIES'] - \
                             mid_price['ROSES'] - DIFFERENCE_MEAN

                worst_bid_price = min(order_depth.buy_orders.keys())
                worst_ask_price = max(order_depth.sell_orders.keys())

                if difference > PERCENT_OF_STD_TO_TRADE_AT * DIFFERENCE_STD:
                    orders = self.calculate_orders(product, order_depth, -int(1e9), worst_bid_price)
                    result[product] = orders
                elif difference < -PERCENT_OF_STD_TO_TRADE_AT * DIFFERENCE_STD:
                    orders = self.calculate_orders(product, order_depth, worst_ask_price, int(1e9))
                    result[product] = orders
            elif product == 'ROSES':
                self.update_buy_sell_roses(state)
                orders = self.trade_roses_with_rhianna(order_depth, product)
                result[product] = orders
            elif product == 'COCONUT':
                option_orders = self.trade_options(state.order_depths)
                for product, orders in option_orders.items():
                    result[product] = orders

        dict = self.cache_results()
        traderData = jsonpickle.encode(dict)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData



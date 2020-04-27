import config
from datetime import datetime
import json
import math
import os
import random
import time
from questrade_api import Questrade

'''
TODO:
- convert _company_check() to decorator
'''

EXPIRY_FORMAT = '%Y-%m-%dT%H:%M:%S.%f%z'

class QuestradeTickerOptions(Questrade):

    def __init__(self, refresh_token = None, auto_refresh = True):
        if auto_refresh:
            if refresh_token is not None:
                raise Exception(
                    ('Provide a refresh token OR request an auto-refresh from '
                     'stored config, NOT both.'))
            with open(os.path.expanduser('~/.questrade.json'), 'r') as QF:
                refresh_token = json.load(QF)['refresh_token']
        if refresh_token is None:
            raise Exception(
                ('Either a valid refresh token or a request for auto-refresh '
                 'are required'))

        super().__init__(refresh_token=refresh_token)
        self.__company_meta = None

    def _overload_robust_request(self, fn, *args, **kwargs):
        '''
        There are times when we might overload the server with requests,
        probably when there are multiple thread running). To combat this, make
        requests robust to server overload
        '''
        result = None

        while True:
            result = fn(*args, **kwargs)
            try:
                if result['code'] == 1006:
                    # Yup, that's some overload
                    time.sleep(random.random())
                    continue
            except:
                # No 'code' key implies that we're good to go
                break

        return result

    def _parse_symbols(self, ticker):
        '''
        Basically a error-checking wrapper for symbols_search, since we still
        need to choose from the companies returned by the API
        '''
        company_meta = None

        companies = self._overload_robust_request(
            self.symbols_search, prefix=ticker.upper() )['symbols']
        for c in companies:
            if c['symbol'] == ticker:
                company_meta = c
                break

        if company_meta is None:
            raise Exception(
                'No company exists with the exact ticker {}'.format(ticker))

        return company_meta

    def _company_check(self):
        if self.__company_meta is None:
            raise Exception(
                ('This operation requires a company to have been loaded via '
                 'load_company().'))

    def load_company(self, ticker):
        # There may be many different objects making requests simultaneously, so
        # we want to attempt to stagger them with ticker-dependent random waits
        random.seed(sum((ord(ch) for ch in ticker)) + time.time())
        self.__company_meta = self._parse_symbols(ticker)


    def get_security_price(self):
        self._company_check()

        response = self._overload_robust_request(
            self.markets_quote, self.__company_meta['symbolId'])

        for info in response['quotes']:
            if info['symbol'] == self.__company_meta['symbol']:
                current_info = info
                break

        if current_info is None:
            raise Exception(
                'Could not find current quote for {}. Exiting.'.format(
                    self.__company_meta['symbol']))

        return current_info['lastTradePrice']

    def get_options(self):
        '''
        return format:
        {
            <expiry>:
                {
                    <symbolId>: {
                        'type': ['C','P'],
                        'strike': <float>,
                        <data>,
                    },
                    ...
                },
                ...
        }
        '''
        self._company_check()

        options = {}

        code = self.__company_meta['symbolId']

        # load up all the available options metadata
        options_meta = self._overload_robust_request(
            self.symbol_options, code)['optionChain']

        for ex in options_meta:
            new_ex = {}
            ex_date = ex['expiryDate']

            # Once we've found the data, we don't care about the minutiea of the
            # timezone and milliseconds
            dir_date = str(datetime.strptime(ex_date, EXPIRY_FORMAT).date())

            # Gather up the series for this expiry.
            quotes = self._overload_robust_request(
                    self.markets_options,
                    filters=[{'underlyingId': code,'expiryDate': ex_date}]
                )['optionQuotes']

            # We reformulate as a dict keyed by the symbolId to make it easier
            # to load in the data in the inner compiling loop below
            series_data = {op['symbolId']: op for op in quotes}

            for s in ex['chainPerRoot'][0]['chainPerStrikePrice']:
                for op_type in ('call', 'put'):
                    op_id = s[op_type + 'SymbolId']
                    new_ex[op_id] = {
                        'type': op_type[0].upper(),
                        'strike': s['strikePrice'],
                        'data': series_data[op_id]
                    }

            options[dir_date] = new_ex

        return options

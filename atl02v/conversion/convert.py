""" Converter class, to convert given values with the 
expressions looked up with the given labels.

Author:

    C.M. Gosmeyer, Aug 2018
"""

from atl02v.conversion.itos import ITOS
from atl02v.conversion.eval_expression import eval_expr

class Converter(object):
    def __init__(self, atl01_labels, atl01_values, channel=None, 
        hkt=None, mnemonic=None, verbose=False):
        """ Converter class.

        Parameters
        ----------
        atl01_labels : list of strings
            The ATL01 full-path labels.
        atl01_values : list of lists
            The ATL01 values.
        channel : int
            The channel number, for forcing conversion lookup.
        hkt : str
            The HKT string, for forcing conversion lookup.
        mnemonic : string
            The ITOS mnemonic, for forcing conversion lookup. 
            If not None, trumps lookup with channel and hkt.
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose
        self.itos = ITOS(verbose=self.verbose)
        self.atl01_labels = atl01_labels
        self.atl01_values = atl01_values
        self.channel = channel
        self.hkt = hkt
        self.mnemonic = mnemonic

    def convert_all(self):
        """ Convert all values for all labels.

        Returns
        -------
        converted_dict : dict
            Keys of ATL01 labels, values of converted values.
        """
        converted_dict = {}
        itos_dict = {}
        for label, values in zip(self.atl01_labels, self.atl01_values):
            if label != None:
                converted_values, itos_values = self.convert(label, values)
                converted_dict[label] = converted_values
                itos_dict[label] = itos_values

        return converted_dict, itos_dict

    def convert(self, label, values):
        """ Convert given values for the given label.

        Parameters
        ----------
        label : str
            The ATL01 label.
        values : list
            The values to be converted from the ATL01 label.

        Returns
        -------
        converted_values : list
            The converted values.
        """
        if self.hkt != None or self.mnemonic != None:
            channel = self.channel
            hkt = self.hkt            
        else:
            descr = self.itos.query4descr(label)
            hkt, channel = self.itos.parse_descr(descr)
        expr = self.itos.query4expr(hkt=hkt, channel=channel, mnemonic=self.mnemonic)

        converted_values = []
        for value in values:
            converted_value = eval_expr(expr, value)
            converted_values.append(converted_value)

        return converted_values, [hkt, channel, expr]


"""Module to read the etu2/ ITOS rec files into a pandas dataframe.

Author:

    C.M. Gosmeyer, Aug 2018
"""

import numpy as np
import os
import pandas as pd 
import xml.etree.cElementTree as et

from atl02v.shared.paths import path_to_itos, itos_xml

class ITOS(object):
    def __init__(self, verbose=True):
        """ ITOS class.

        Parameters
        ----------
        verbose : {True, False}
            False by default.
        """
        self.verbose = verbose

        # Read xml of the ITOS database.
        xml_file = os.path.join(path_to_itos, itos_xml)
        self.idf = parse_xml_file(xml_file)

        # Set the label2description dataframe
        label2descr_file = os.path.join(path_to_itos, 'ATL01_Label2Description.csv')
        self.label2descr = pd.read_csv(label2descr_file)

    def query4descr(self, label):
        """ Use ATL01 label to query 'ATL01_Label2Description.csv' for description.

        Parameters
        ----------
        label : string
            The ATL01 full-path label.

        Returns
        -------
        descr : string
            The description looked up for the ATL01 label.
        """
        # Clean up label so any group names not included
        label = label.split('/')[-1]
        descr_raw = self.label2descr['Description'][self.label2descr['Label'] == label]
        descr = descr_raw[descr_raw.index[-1]]
        if self.verbose:
            print(" ")
            print("query4descr")
            print("label: ", label)
            print("descr_raw: ", descr_raw)
            print("descr: ", descr)

        return descr

    def parse_descr(self, descr):
        """ Parse the description for channel and HKT.

        Parameters
        ----------
        descr : string
            The description looked up for the ATL01 label.

        Returns
        -------
        hkt : string
            The HKT string, for example, 'A_HKT_A'. 
            'None' if the parse unsucessful.
        channel : string
            The channel number. 'None' if the parse unsucessful.
        """
        last_word = descr.split(' ')[-1]

        if self.verbose:
            print(" ")
            print("parse_descr")            
            print("last_word: ", last_word)

        if 'HKT' in last_word:
            hkt = last_word.split('.')[0]
            channel = last_word.split('[')[-1].strip(']')
            if self.verbose:
                print("hkt, channel: ", hkt, channel)           
            return hkt, channel
        else:
            ## actually need do something else?
            print("WARNING: Could not find HKT and channel in '{}'".format(descr))
            return None, None

    def query4expr(self, hkt, channel, mnemonic=None):
        """ Use HKT and channel to query self.idf for the expression.

        Parameters
        ----------
        hkt : string
            The HKT string, for example, 'A_HKT_A'. 
        channel : string
            The channel number.
        mnemonic : string
            The ITOS mnemonic. If not None, trumps lookup with channel and hkt.

        Returns
        -------
        expr : str
            The expression looked up with HKT and channel.
            'None' if look up unsucessful.      
        """
        if self.verbose:
            print(" ")
            print("query4expr")

        try:
            if mnemonic != None:
                if self.verbose:
                    print("mnemonic: ", mnemonic)
                conv_name = self.idf['ConversionName'][self.idf['Mnemonic'] == mnemonic].values[0]
                if self.verbose:
                    print("conv_name: ", conv_name)
                expr_raw = self.idf['Expression'][self.idf['ConversionName'] == conv_name][self.idf['Mnemonic'].isnull()]
            elif 'HKT' in str(hkt) and channel != None:
                expr_raw = self.idf['Expression'][self.idf['HKT'] == hkt][self.idf['Channel'] == channel]
            elif ('HKT' in str(hkt) and channel == None) or 'MVOLTS' in hkt:
                expr_raw = self.idf['Expression'][self.idf['HKT'] == hkt]
            else:
                expr_raw = self.idf['Expression']

            if self.verbose:
                print("expr_raw: ", expr_raw)

            # Retrieve the expression.
            # This also ensures that if there are more than one expression, only the 
            # latest will be used.
            expr = expr_raw[expr_raw.index[-1]]
            if self.verbose:
                print("try - expr: ", expr)

            try:
                if expr[0:3] == '( (':
                    expr = expr[1:-1]
                    if self.verbose:
                        print("expr: ", expr)
            except:
                if expr == None:
                    # If first pass fails, then search via a MnemonicName.
                    conv = self.idf['ConversionName'][self.idf['HKT'] == hkt][self.idf['Channel'] == channel]  
                    expr_raw = self.idf['Expression'][self.idf['ConversionName'] == conv[conv.index[-1]]][self.idf['Mnemonic'].isnull()]
                    expr = expr_raw[expr_raw.index[-1]]
                    if self.verbose:
                        print("except - expr: ", expr)
            return expr
        except:
            print("ERROR: Could not find expression for {}, {}".format(hkt, channel))
            return None

def getvalueofnode(node):
    """ Return node text or None 

    References
    ----------
    http://gokhanatil.com/2017/11/python-for-data-science-importing-xml-to-pandas-dataframe.html
    """
    return node.text if node is not None else None


def parse_xml_file(filename):
    """ Parse the ITOS XML file for 'ExpressionAlgorithm' and 
    'ExpressionConversion' nodes, and 'Mnemonic' nodes if they contain
    channel numbers.

    Parameters
    ----------
    filename : string
        The name of the XML file.

    Returns
    -------
    df : pandas DataFrame
        The DataFrame containing parsed ITOS XML database.
    """
    conversion_names = []
    coeffs = []
    expressions = []
    channels = []
    hkts = []
    mnemonics = []

    parsed_xml = et.parse(filename)

    for node in parsed_xml.getroot():

        if node.tag == 'ExpressionAlgorithm' or node.tag == 'ExpressionConversion':
            conversion_name = node.attrib.get('name')
            expression = getvalueofnode(node.find("expression"))

            coeffs.append(None)
            conversion_names.append(conversion_name)
            expressions.append(expression)
            mnemonics.append(None)
            if '_' in conversion_name:
                hkts.append('_'.join(conversion_name.split('_chan_')[0].split('_')[1:]))
            else:
                hkts.append(conversion_name)

            if 'chan_' in conversion_name:
                channels.append(conversion_name.split('chan_')[-1])
            else:
                channels.append(None)

        # Because many conversions link to same expression via mnemonics.
        elif node.tag == 'Mnemonic':
            sourcefield = getvalueofnode(node.find("sourceFields/sourceField"))
            if sourcefield != None and '.chan' in sourcefield and 'chan[' not in node.attrib.get('name'):
                mnemonics.append(node.attrib.get('name'))
                conversion_name = getvalueofnode(node.find("conversion"))
                conversion_names.append(conversion_name)
                coeffs.append(None)
                expressions.append(None)
                hkts.append(sourcefield.split('.')[0])
                channels.append(sourcefield.split('[')[1].strip(']'))

            elif sourcefield != None and ('LRS_HK_' in node.attrib.get('name')) and \
                    ('TEMP' not in node.attrib.get('name')):
                mnemonics.append(node.attrib.get('name'))
                conversion_name = getvalueofnode(node.find("conversion"))
                conversion_names.append(conversion_name)
                coeffs.append(None)
                expressions.append(None)  
                hkts.append(sourcefield.split('.')[0])
                channels.append(None)

            elif sourcefield != None and ('LRS_HK_' in node.attrib.get('name')) and \
                    ('TEMP' in node.attrib.get('name')):
                mnemonics.append(node.attrib.get('name'))
                conversion_name = sourcefield
                conversion_names.append(conversion_name)
                coeffs.append(None)
                expressions.append(None)  
                hkts.append(sourcefield.split('.')[0])
                channels.append(None)

        elif node.tag == 'PolynomialConversion':
            if 'A_LRS_HK' in node.attrib.get('name'):
                # Capture the coefficients and build expression
                coeff = [node[0][0].text, node[0][1].text]
                coeffs.append(coeff)
                expression = '{} + x*{}'.format(coeff[0], coeff[1])
                expressions.append(expression)
                mnemonics.append(None)
                conversion_name = node.attrib.get('name')
                conversion_names.append(conversion_name)
                hkts.append('A_LRS_HK')
                channels.append(None)


    # Create dataframe
    df = pd.DataFrame({'ConversionName':conversion_names, 'HKT':hkts, 
        'Channel':channels, 'Expression':expressions, 'Coeffs':coeffs, 
        'Mnemonic':mnemonics}) 

    return df



LABEL_KEY = "T"
FEATURE_KEY = [
 'CO(GT)',
 'PT08_S1(CO)',
 'NMHC(GT)',
 'C6H6(GT)',
 'PT08_S2(NMHC)',
 'NOx(GT)',
 'PT08_S3(NOx)',
 'NO2(GT)',
 'PT08_S4(NO2)',
 'PT08_S5(O3)',
 'RH',
 'AH']

ALL_KEY = [
 'CO(GT)',
 'PT08_S1(CO)',
 'NMHC(GT)',
 'C6H6(GT)',
 'PT08_S2(NMHC)',
 'NOx(GT)',
 'PT08_S3(NOx)',
 'NO2(GT)',
 'PT08_S4(NO2)',
 'PT08_S5(O3)',
 'RH',
 'AH',
 'T']

def transformed_name(key):
    """Rename transformed features"""

    return key + "_tn"

def vectorize_name(key):
    """"Rename vectorize features"""

    return key + "_Vect"

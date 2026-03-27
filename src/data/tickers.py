from random import * 

TICKERS = """ABIO
ABRD
AFKS
AFLT
AGRO
AKRN
ALRS
AMEZ
APRI
APTK
AQUA
ARSA
ASSB
ASTR
AVAN
BANE
BANEP
BELU
BISVP
BRZL
BSPB
BSPBP
CARM
CBOM
CHGZ
CHKZ
CHMF
CHMK
CIAN
CNTL
CNTLP
DATA
DELI
DIAS
DIOD
DVEC
DZRD
DZRDP
EELT
ELFV
ELMT
ENPG
ETLN
EUTR
FEES
FESH
FIXP
FLOT
GAZP
GCHE
GECO
GEMA
GEMC
GLTR
GMKN
HEAD
HIMCP
HNFG
HYDR
IGST
IGSTP
INGR
IRAO
IRKT
IVAT
JNOS
JNOSP
KAZT
KAZTP
KBSB
KCHE
KCHEP
KGKC
KGKCP
KLSB
KLVZ
KMAZ
KMEZ
KOGK
KRKN
KRKNP
KRKOP
KROT
KROTP
KRSB
KRSBP
KUBE
KUZB
KZOS
KZOSP
LEAS
LENT
LKOH
LNZL
LNZLP
LSNG
LSNGP
LSRG
LVHK
MAGE
MAGEP
MAGN
MBNK
MDMG
MFGS
MFGSP
MGKL
MGNT
MGTS
MGTSP
MISB
MISBP
MOEX
MRKC
MRKK
MRKP
MRKS
MRKU
MRKV
MRKY
MRKZ
MRSB
MSNG
MSRS
MSTT
MTLR
MTLRP
MTSS
MVID
NAUK
NFAZ
NKHP
NKNC
NKNCP
NKSH
NLMK
NMTP
NNSB
NNSBP
NSVZ
NVTK
OGKB
OKEY
OMZZP
OZON
PHOR
PIKK
PLZL
PMSB
PMSBP
POSI
PRFN
PRMB
PRMD
QIWI
RASP
RDRB
RENI
RGSS
RKKE
RNFT
ROLO
ROSB
ROSN
ROST
RTGZ
RTKM
RTKMP
RTSB
RTSBP
RUAL
RUSI
RZSB
SAGO
SAGOP
SARE
SAREP
SBER
SBERP
SELG
SFIN
SGZH
SIBN
SLEN
SMLT
SNGS
SNGSP
SOFL
SPBE
STSB
STSBP
SVAV
SVCB
SVET
SVETP
TASB
TASBP
TATN
TATNP
TCSG
TGKA
TGKB
TGKBP
TGKN
TNSE
TORS
TORSP
TRMK
TRNFP
TTLK
TUZA
UGLD
UKUZ
UNAC
UNKL
UPRO
URKZ
USBN
UTAR
VEON-RX
VGSB
VGSBP
VJGZ
VJGZP
VKCO
VLHZ
VRSB
VRSBP
VSEH
VSMO
VSYD
VSYDP
VTBR
WTCM
WTCMP
WUSH
YAKG
YDEX
YKEN
YKENP
YRSB
YRSBP
ZAYM
ZILL
ZVEZ
ABIO
ABRD
AFKS
AFLT
AGRO
AKRN
ALRS
AMEZ
APRI
APTK
AQUA
ARSA
ASSB
ASTR
AVAN""".split("\n")
TICKERS = sorted(list(set(TICKERS)))
print(len(TICKERS))

data = TICKERS[:]

seed(42)
shuffle(data)



a = data[0:30]
b = data[30:60]
c = data[60:90]

with open("tickers_and.txt", 'w') as file:
    txt = "\n".join(a)
    file.write(txt)

with open("tickers_bor.txt", 'w') as file:
    txt = "\n".join(b)
    file.write(txt)

with open("tickers_dem.txt", 'w') as file:
    txt = "\n".join(c)
    file.write(txt)


# ── процент пропусков по тикерам (относительно эталонного торгового календаря) ──

DAILY_MISSING_PCT = {
    "ABIO": 0.49,
    "ABRD": 10.82,
    "AFKS": 0.0,
    "AFLT": 0.0,
    "AKRN": 0.69,
    "ALRS": 0.0,
    "AMEZ": 11.62,
    "APTK": 0.42,
    "ARSA": 7.01,
    "ASSB": 7.56,
    "AVAN": 29.52,
    "BANE": 0.28,
    "BANEP": 0.28,
    "BISVP": 11.72,
    "BLNG": 0.97,
    "BRZL": 5.48,
    "BSPB": 0.21,
    "CHGZ": 31.7,
    "CHKZ": 25.84,
    "CHMF": 0.0,
    "CHMK": 0.97,
    "CNTL": 5.83,
    "CNTLP": 3.36,
    "DIOD": 1.94,
    "DVEC": 1.91,
    "DZRD": 13.39,
    "DZRDP": 5.24,
    "FEES": 0.49,
    "FESH": 0.42,
    "GAZA": 3.64,
    "GAZAP": 14.19,
    "GAZC": 99.86,
    "GAZP": 0.0,
    "GAZS": 99.9,
    "GAZT": 99.86,
    "GCHE": 0.76,
    "GMKN": 0.14,
    "HIMCP": 8.36,
    "HYDR": 0.0,
    "IGST": 11.1,
    "IGSTP": 18.14,
    "IRAO": 0.42,
    "IRKT": 0.21,
    "JNOS": 16.79,
    "JNOSP": 12.59,
    "KAZT": 12.0,
    "KAZTP": 17.07,
    "KBSB": 18.35,
    "KCHE": 12.87,
    "KCHEP": 26.43,
    "KGKC": 44.92,
    "KGKCP": 45.09,
    "KLSB": 12.94,
    "KMAZ": 0.21,
    "KMEZ": 3.82,
    "KOGK": 22.96,
    "KRKN": 24.9,
    "KRKNP": 4.2,
    "KRKOP": 11.55,
    "KROT": 4.93,
    "KROTP": 6.73,
    "KRSB": 9.33,
    "KRSBP": 8.5,
    "KUZB": 5.65,
    "KZOS": 1.94,
    "KZOSP": 3.16,
    "LIFE": 2.98,
    "LKOH": 0.0,
    "LNZL": 1.11,
    "LNZLP": 1.01,
    "LPSB": 30.59,
    "LSNG": 2.43,
    "LSNGP": 2.25,
    "LSRG": 0.35,
    "LVHK": 10.68,
    "MAGE": 11.52,
    "MAGEP": 12.0,
    "MAGN": 0.0,
    "MFGS": 12.42,
    "MFGSP": 6.83,
    "MGNT": 0.0,
    "MGNZ": 39.99,
    "MGTS": 10.09,
    "MGTSP": 3.05,
    "MISB": 30.39,
    "MISBP": 30.66,
    "MOEX": 0.0,
    "MRKC": 0.62,
    "MRKK": 1.49,
    "MRKP": 0.62,
    "MRKS": 1.42,
    "MRKU": 0.52,
    "MRKV": 0.62,
    "MRKY": 1.01,
    "MRKZ": 0.62,
    "MRSB": 6.87,
    "MSNG": 0.07,
    "MSRS": 0.42,
    "MSTT": 1.35,
    "MTLR": 0.07,
    "MTLRP": 0.07,
    "MTSS": 0.03,
    "MVID": 0.21,
    "NAUK": 11.52,
    "NFAZ": 11.27,
    "NKNC": 1.08,
    "NKNCP": 0.97,
    "NKSH": 10.82,
    "NLMK": 0.0,
    "NMTP": 0.28,
    "NNSB": 27.51,
    "NNSBP": 16.48,
    "NSVZ": 14.57,
    "NVTK": 0.14,
    "OGKB": 0.21,
    "OMZZP": 13.7,
    "PAZA": 28.75,
    "PHOR": 0.07,
    "PIKK": 0.0,
    "PLZL": 0.24,
    "PMSB": 6.69,
    "PMSBP": 1.77,
    "PRFN": 6.28,
    "PRMB": 39.82,
    "RASP": 0.21,
    "RBCM": 0.97,
    "RDRB": 44.75,
    "RGSS": 8.01,
    "RKKE": 5.72,
    "ROLO": 5.79,
    "ROSN": 0.0,
    "ROST": 3.05,
    "RTGZ": 21.23,
    "RTKM": 0.0,
    "RTKMP": 0.21,
    "RTSB": 25.46,
    "RTSBP": 16.82,
    "RUSI": 34.51,
    "RZSB": 11.79,
    "SAGO": 13.25,
    "SAGOP": 12.97,
    "SARE": 4.23,
    "SAREP": 4.65,
    "SBER": 0.0,
    "SBERP": 0.0,
    "SELG": 0.14,
    "SIBN": 0.28,
    "SNGS": 0.0,
    "SNGSP": 0.0,
    "STSB": 2.57,
    "STSBP": 1.87,
    "SVAV": 0.21,
    "TASB": 17.03,
    "TASBP": 14.81,
    "TATN": 0.07,
    "TATNP": 0.07,
    "TGKA": 0.21,
    "TGKB": 0.97,
    "TGKBP": 5.69,
    "TGKN": 2.46,
    "TORS": 21.37,
    "TORSP": 6.56,
    "TRMK": 0.21,
    "TRNFP": 0.14,
    "TTLK": 2.22,
    "TUZA": 15.82,
    "UKUZ": 5.1,
    "UNAC": 0.35,
    "UNKL": 9.19,
    "URKZ": 12.76,
    "USBN": 2.46,
    "UTAR": 1.77,
    "VGSB": 12.76,
    "VGSBP": 14.53,
    "VJGZ": 24.56,
    "VJGZP": 21.75,
    "VLHZ": 3.47,
    "VRSB": 25.56,
    "VRSBP": 27.09,
    "VSMO": 0.97,
    "VSYD": 22.1,
    "VSYDP": 29.8,
    "VTBR": 0.14,
    "WTCM": 8.57,
    "WTCMP": 20.57,
    "YAKG": 14.88,
    "YKEN": 4.82,
    "YKENP": 8.6,
    "YRSB": 37.53,
    "YRSBP": 32.36,
    "ZILL": 5.24,
    "ZVEZ": 9.92,
}

HOURLY_MISSING_PCT = {
    "ABIO": 27.62,
    "ABRD": 42.86,
    "AFKS": 0.88,
    "AFLT": 0.15,
    "AKRN": 21.57,
    "ALRS": 0.11,
    "AMEZ": 39.29,
    "APTK": 19.28,
    "ARSA": 55.46,
    "ASSB": 49.46,
    "AVAN": 54.84,
    "BANE": 14.65,
    "BANEP": 13.53,
    "BISVP": 53.87,
    "BLNG": 28.64,
    "BRZL": 41.5,
    "BSPB": 13.17,
    "CHGZ": 64.02,
    "CHKZ": 69.62,
    "CHMF": 0.08,
    "CHMK": 33.33,
    "CNTL": 43.06,
    "CNTLP": 38.06,
    "DIOD": 35.55,
    "DVEC": 34.21,
    "DZRD": 64.77,
    "DZRDP": 53.85,
    "FEES": 1.06,
    "FESH": 14.38,
    "GAZA": 52.95,
    "GAZAP": 62.81,
    "GAZC": 99.99,
    "GAZP": 0.0,
    "GAZS": 99.99,
    "GAZT": 99.99,
    "GCHE": 23.46,
    "GMKN": 0.19,
    "HIMCP": 44.29,
    "HYDR": 0.24,
    "IGST": 55.83,
    "IGSTP": 59.3,
    "IRAO": 0.74,
    "IRKT": 19.2,
    "JNOS": 64.07,
    "JNOSP": 60.96,
    "KAZT": 46.52,
    "KAZTP": 52.56,
    "KBSB": 61.27,
    "KCHE": 62.73,
    "KCHEP": 75.48,
    "KGKC": 70.86,
    "KGKCP": 67.62,
    "KLSB": 44.61,
    "KMAZ": 19.28,
    "KMEZ": 45.06,
    "KOGK": 73.58,
    "KRKN": 75.08,
    "KRKNP": 37.52,
    "KRKOP": 58.15,
    "KROT": 43.58,
    "KROTP": 56.28,
    "KRSB": 52.2,
    "KRSBP": 53.77,
    "KUZB": 36.73,
    "KZOS": 34.01,
    "KZOSP": 36.38,
    "LIFE": 35.1,
    "LKOH": 0.02,
    "LNZL": 32.32,
    "LNZLP": 30.19,
    "LPSB": 60.55,
    "LSNG": 24.95,
    "LSNGP": 24.22,
    "LSRG": 13.08,
    "LVHK": 51.69,
    "MAGE": 54.63,
    "MAGEP": 54.18,
    "MAGN": 0.58,
    "MFGS": 65.07,
    "MFGSP": 62.61,
    "MGNT": 0.16,
    "MGNZ": 77.02,
    "MGTS": 56.09,
    "MGTSP": 33.98,
    "MISB": 66.11,
    "MISBP": 65.37,
    "MOEX": 0.07,
    "MRKC": 20.46,
    "MRKK": 35.5,
    "MRKP": 21.84,
    "MRKS": 34.76,
    "MRKU": 27.03,
    "MRKV": 21.29,
    "MRKY": 27.75,
    "MRKZ": 27.47,
    "MRSB": 51.24,
    "MSNG": 12.61,
    "MSRS": 22.35,
    "MSTT": 23.68,
    "MTLR": 8.36,
    "MTLRP": 9.72,
    "MTSS": 0.2,
    "MVID": 10.88,
    "NAUK": 51.14,
    "NFAZ": 51.47,
    "NKNC": 31.16,
    "NKNCP": 23.47,
    "NKSH": 55.72,
    "NLMK": 0.12,
    "NMTP": 14.27,
    "NNSB": 67.05,
    "NNSBP": 63.22,
    "NSVZ": 47.06,
    "NVTK": 0.35,
    "OGKB": 9.97,
    "OMZZP": 62.67,
    "PAZA": 67.89,
    "PHOR": 2.11,
    "PIKK": 3.96,
    "PLZL": 1.71,
    "PMSB": 43.06,
    "PMSBP": 35.84,
    "PRFN": 30.52,
    "PRMB": 81.7,
    "RASP": 13.07,
    "RBCM": 24.42,
    "RDRB": 65.78,
    "RGSS": 37.02,
    "RKKE": 44.9,
    "ROLO": 30.62,
    "ROSN": 0.03,
    "ROST": 49.55,
    "RTGZ": 79.25,
    "RTKM": 0.89,
    "RTKMP": 14.76,
    "RTSB": 62.98,
    "RTSBP": 62.3,
    "RUSI": 61.64,
    "RZSB": 53.32,
    "SAGO": 59.88,
    "SAGOP": 63.51,
    "SARE": 53.54,
    "SAREP": 54.98,
    "SBER": 0.01,
    "SBERP": 0.14,
    "SELG": 21.18,
    "SIBN": 9.25,
    "SNGS": 0.24,
    "SNGSP": 0.14,
    "STSB": 45.42,
    "STSBP": 38.39,
    "SVAV": 20.72,
    "TASB": 62.05,
    "TASBP": 66.02,
    "TATN": 0.26,
    "TATNP": 1.67,
    "TGKA": 14.81,
    "TGKB": 29.62,
    "TGKBP": 44.12,
    "TGKN": 39.95,
    "TORS": 71.18,
    "TORSP": 60.45,
    "TRMK": 12.96,
    "TRNFP": 3.27,
    "TTLK": 35.56,
    "TUZA": 52.67,
    "UKUZ": 55.07,
    "UNAC": 18.22,
    "UNKL": 47.74,
    "URKZ": 69.52,
    "USBN": 34.46,
    "UTAR": 38.12,
    "VGSB": 63.87,
    "VGSBP": 64.6,
    "VJGZ": 60.55,
    "VJGZP": 60.0,
    "VLHZ": 39.7,
    "VRSB": 53.84,
    "VRSBP": 63.17,
    "VSMO": 20.75,
    "VSYD": 69.32,
    "VSYDP": 74.05,
    "VTBR": 0.21,
    "WTCM": 54.63,
    "WTCMP": 58.17,
    "YAKG": 47.67,
    "YKEN": 53.48,
    "YKENP": 62.54,
    "YRSB": 74.58,
    "YRSBP": 70.2,
    "ZILL": 51.04,
    "ZVEZ": 53.71,
}

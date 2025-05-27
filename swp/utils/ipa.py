pulmonic_consonants = {
    "\u0070",
    "\u0062",
    "\u006d",
    "\u0299",
    "\u0278",
    "\u03b2",
    "\u0271",
    "\u2c71",  # (F25F)
    "\u0066",
    "\u0076",
    "\u028b",
    "\u0074",
    "\u0064",
    "\u006e",
    "\u0072",
    "\u027e",
    "\u03b8",
    "\u00f0",
    "\u0073",
    "\u007a",
    "\u0283",
    "\u0292",
    "\u026c",
    "\u026e",
    "\u0279",
    "\u006c",
    "\u0288",
    "\u0256",
    "\u0273",
    "\u027d",
    "\u0282",
    "\u0290",
    "\u027b",
    "\u026d",
    "\u0063",
    "\u025f",
    "\u0272",
    "\u00e7",
    "\u029d",
    "\u006a",
    "\u028e",
    "\u006b",
    "\u0261",  # (0067)
    "\u014b",
    "\u0078",
    "\u0263",
    "\u0270",
    "\u029f",
    "\u0071",
    "\u0262",
    "\u0274",
    "\u0280",
    "\u03c7",
    "\u0281",
    "\u0127",
    "\u0295",
    "\u0294",
    "\u0068",
    "\u0266",
}

non_pulmonic_consonants = {
    "\u0298",
    "\u01c0",
    "\u01c3",
    "\u01c2",
    "\u01c1",
    "\u0253",
    "\u0257",
    "\u0284",
    "\u0260",
    "\u029b",
    "\u02bc",
}

other_symbols = {
    "\u028d",
    "\u0077",
    "\u0265",
    "\u029c",
    "\u02a2",
    "\u02a1",
    "\u0255",
    "\u0291",
    "\u027a",
    "\u0267",
    "\u0361",  # combines two symbols
    "\u035c",  # combines two symbols
}

vowels = {
    "\u0069",
    "\u0079",
    "\u026a",
    "\u028f",
    "\u0065",
    "\u00f8",
    "\u025b",
    "\u0153",
    "\u00e6",
    "\u0061",
    "\u0276",
    "\u0268",
    "\u0289",
    "\u0258",
    "\u0275",
    "\u0259",
    "\u025c",
    "\u025e",
    "\u0250",
    "\u026f",
    "\u0075",
    "\u028a",
    "\u0264",
    "\u006f",
    "\u028c",
    "\u0254",
    "\u0251",
    "\u0252",
}

diacritics = {
    "\u0325",
    "\u030a",
    "\u032c",
    "\u02b0",
    "\u0339",
    "\u031c",
    "\u031f",
    "\u0320",
    "\u0308",
    "\u033d",
    "\u0329",
    "\u032f",
    "\u02de",
    "\u0324",
    "\u0330",
    "\u032b",
    "\u02b7",
    "\u02b2",
    "\u02e0",
    "\u02e4",
    "\u0334",
    "\u031d",
    "\u031e",
    "\u0318",
    "\u0319",
    "\u032a",
    "\u033a",
    "\u033b",
    "\u0303",
    "\u207f",
    "\u02e1",
    "\u031a",  # 02FA
}

suprasegmentals = {
    "\u02c8",
    "\u02cc",
    "\u02d0",
    "\u02d1",
    "\u0306",
    "\u007c",
    "\u2016",  # 007C + 007C
    "\u002e",
    "\u203f",
}

tones_and_word_accents = {
    "\u030b",
    "\u02e5",
    "\u0301",
    "\u02e6",
    "\u0304",
    "\u02e7",
    "\u0300",
    "\u02e8",
    "\u030f",
    "\u02e9",
    "\u030c",
    "\u02e9\u02e5",
    "\u0302",
    "\u02e5\u02e9",
    "\u1dc4",
    "\u02e7\u02e5",
    "\u1dc5",
    "\u02e9\u02e7",
    "\u1dc8",
    "\u02e7\u02e6\u02e8",
    "\ua71c",
    "\ua71b",
    "\u2197",
    "\u2198",
}

# in_unicode_but_not_on_the_ipa_chart = {
#     "\u025A",  # 0259 + 02DE
#     "\u025D",  # 025C + 02DE
#     "\u02A3",  # 0064 + 0361 + 007A
#     "\u02A4",  # 0064 + 0361 + 0292
#     "\u02A5",  # 0064 + 0361 + 0291
#     "\u02A6",  # 0074 + 0361 + 0073
#     "\u02A7",  # 0074 + 0361 + 0283
#     "\u02A8",  # 0074 + 0361 + 0255
#     "\u026B",
#     "\u02B1",  # 0324
#     "\u02B3",
#     "\u02B4",
#     "\u02B5",
#     "\u02B6",
#     "\u02C0",  # 0330
#     "\u0322",
#     "\u1DC6",
#     "\u1DC7",
#     "\u1DC9",
# }

# alternatives = {
#     "\u02FA": "\u031A",  # end high tone instead of combining left angle above
#     "\u0067": "\u0261",  # normal g instead of script g
#     "\uF25F": "\u2C71",  # v with right hook in certain fonts
#     "\u007C\u007C": "\u2016",  # vertical line twice instead of double vertical line
#     "\u003A": "\u02D0",  # colon instead of modifier triangular column
#     "\u0021": "\u01C3",  # exclamation mark instead of retroflex click
#     "\u025A": "\u0259\u02DE",  # schwa with hook instead of schwa + rhotic hook
#     "\u025D": "\u025C\u02DE",  # reversed open e with hook instead of reversed open e + rhotic hook
#     "\u02A3": "\u0064\u0361\u007A",  # affricate
#     "\u02A4": "\u0064\u0361\u0292",  # affricate
#     "\u02A5": "\u0064\u0361\u0291",  # affricate
#     "\u02A6": "\u0074\u0361\u0073",  # affricate
#     "\u02A7": "\u0074\u0361\u0283",  # affricate
#     "\u02A8": "\u0074\u0361\u0255",  # affricate
#     "\u03B5": "\u025B",  # epsilon instead of open e
#     "\u01DD": "\u0259",  # turned e instead of schwa
#     "\u026B": "\u006C\u0334",  # L with Middle Tilde instead of L + combining tilde overlay
#     "\u200D": "\u035C",  # used in en_UK.txt for tied phonemes
#     "\u0020": "",  # blank space for word separation
#     "\u00B2": "",  # is before most sv.txt pronunciation
#     "\u2040": "\u203F",  # used once in de.txt for linking
#     "\u0030\u0072": "\u0072\u0325",  # 0 is used in is.txt instead of voiceless diacritic
#     "\u0030": "\u0072\u0325",  # þverfaglegt is missing the r in is.txt
#     "\u0023": "\u002E",  # seems to be used for non diphtong sounds in is.txt
#     "\u0027": "\u02C8",  # using apostrophe instead of primary stress
#     "\u030D": "\u0329",  # vertical line above instead of vertical line under, normal for some letters
#     "\u2193": "\uA71C",  # downwards arrow instead of raised down arrow
#     "\u2191": "\uA71B",  # upwards arrow instead of raised up arrow
#     "\u1EA1": "\u0061",  # Naz in de.txt
#     "\u005F": "\u0063",  # _ instead of c in is.txt
#     "\u002D": "",  # word separator in multiple languages
#     "\u007E": "",  # framtíðarhorfur in is.txt
#     "\u2014": "",  # in der Pipeline in de.txt
#     "\u003F": "",  # before some pronunciation in sv.txt + เญียง in tts.txt
#     "\u0311": "\u032F",  # combining inverted breve above instead of combining inverted breve under, normal for some letters
#     "\u02B1": "\u0324",  # Modifier small h with hook for breathy voiced instead of combining diaresis below
#     "\u02C0": "\u0330",  # modifier glottal stop for creaky voiced instead of combining tilde below
#     "\u0348": "",  # non IPA, used in ko.txt for tensed consonants/faucalized voice
#     "\u1d50\u0253": "\u006D\u0361\u0253",  # prenasalization
#     "\u1d50\u0076": "\u006D\u0361\u0076",  # prenasalization
#     "\u1D51\u0261": "\u014B\u0361\u0261",  # prenasalization
# }

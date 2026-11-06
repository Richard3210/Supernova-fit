def prior(Om, Ol, Ok):
    if Om > 0 and (1-Om-Ol-Ok) > 0:
        return 1
    else:
        return 0
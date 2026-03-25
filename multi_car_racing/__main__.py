try:
    from .multi_car_racing import main
    main()
except Exception:
    import runpy
    runpy.run_module(f"{__package__}.multi_car_racing", run_name="__main__")

from src.display import display


def main():

    the_dataset = 'data/w8-8_s9_c8_0_100_1.csv'
    the_model = 'models/nff_h32_lr2_e1000.csv'

    the_display = display.Display(the_dataset, the_model)
    the_display.root.mainloop()


main()





from src.display import display


def main():

    the_dataset = 'data/w6-6_s9_c8_0_5_0.csv'
    the_model = 'models/nff_h32_lr1_e1000.csv'

    the_display = display.Display(the_dataset, the_model)
    the_display.root.mainloop()


main()





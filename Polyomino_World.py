from src.display import display


def main():

    the_dataset = 'data/w8-8_s3_c2_0_10_5.csv'
    the_model = 'models/nff_h32_lr200_e1000_wn7.csv'

    the_display = display.Display(the_dataset, the_model)
    the_display.root.mainloop()


main()

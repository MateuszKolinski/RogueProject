# Unnamed Rogue Card Game
# Created by Mateusz Kolinski MateuszPKolinski@gmail.com

### Still in progress
### Currently finishing card generation

# Card generation step-by-step

To generate a card, we need:
- three template images (pictures below),
- one image depicting a desired character
- a database entry specifying said character's name, stats and colors

| | | |
|:-----------------------:|:-------------------------:|:-------------------------:|
| <img width="200" height="280" alt="White Template" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/Templates/WhiteTemplate.png"> White Card Template |  <img width="200" height="280" alt="Sparks image" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/Templates/Web.png"> Border Fractals | <img width="200" height="280" alt="Border Template" src="https://github.com/MateuszKolinski/RogueProject/blob/main/Templates/InnerBorderTemplate.png?raw=true"> Border Template |

First step of card generation is to divide fractal image into border fractal and inner fractal images. They will have different opacity going forward.

<img width="200" height="280" alt="Fractal outer" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/sparks_image_outer.png">

<img width="200" height="280" alt="Fractal inner" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/sparks_image_inner.png">

Then we determine the colors of the card. Any card can currently have one, two or three colors. 

# One color card

One color card is created by simply changing all non-black pixels in White Card Template to the desired color.

<img width="200" height="280" alt="One color" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/1color.png">

# Two color card

To generate a card with two colors, card is split vertically in two. Each slice is colored just like in one color card example. Then both slices are joined together. Finally, to make the color transition smooth, a simple gradient is created between the two colors.

<img width="200" height="280" alt="2color" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/2color.png>

# Three color card

To generate a card with three colors, the card space is divided into two trapezes and one triangle. Those slices are colored just like in previous examples. Finally, to achieve a smooth color transition, three gradients are created - one vertical and two angled according to the triangle's angles. 

<img width="200" height="280" alt="3color" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/3color.png>

After generating a colored card template, two previously discussed fractal images are weight-blended onto it.

<img width="200" height="280" alt="sparky" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/card_colour_image_with_web.png>

Next step is to add card borders from the Border Template onto character image.

<img width="200" height="280" alt="creature_n_border" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/creature_n_border.png>

Then we merge colored card template with character and border template.

<img width="200" height="280" alt="reature_n_border_n_colour" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/creature_n_border_n_colour.png>

Finally, any text describing character's name and stats is added on top of the card.

<img width="200" height="280" alt="Final" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/final.png>

Lastly, allegience logos are added at the bottom of the card. Each logo is colored according to its color scheme.

<img width="200" height="280" alt="Final Logos" src=https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/assets/final.png>

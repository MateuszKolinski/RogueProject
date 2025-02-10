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
|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="200" height="280" alt="White Template" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/Templates/WhiteTemplate.png"> White Card Template |  <img width="200" height="280" alt="Sparks image" src="https://raw.githubusercontent.com/MateuszKolinski/RogueProject/refs/heads/main/Templates/Web.png"> Border fractals | <img width="200" height="280" alt="Border Template" src="https://github.com/MateuszKolinski/RogueProject/blob/main/Templates/InnerBorderTemplate.png?raw=true"> Border Template |

First step after loading the above images is to determine the colors of the card. Any card can currently have one, two or three colors. 

# One color card

One color card is created by simply changing all non-black pixels in White Card Template to the desired color and then applying a weighted blend with white image which has its opacity turned according to the desired color scheme.

## Two color card

To generate a card with two colors, card is split vertically in two. Each slice is colored just like in one color card example. Then both slices are joined together. Finally, to make the color transition smooth, a simple gradient is created between the two colors.





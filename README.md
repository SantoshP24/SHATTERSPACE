
# SHATTERSPACE

**Deep space survival, with a pilot that flies itself.**

SHATTERSPACE is a space survival game where you (or an AI) fly a ship through asteroid fields and try to last as long as possible. Dodge what you can, shoot what you can't, and watch your health.

What makes it a bit different is that the ship can fly itself. Switch on the autopilot and it decides where to move, what to shoot, and when to run.

## Play it

**[Play SHATTERSPACE in your browser](https://shatterspace.vercel.app/)**


## What you get

- **Two modes:** Survival and Attack
- **Five difficulty levels,** from Cadet up to Nightmare
- **Score, health, and escalating waves** to keep things tense
- **Keyboard controls** on desktop and **touch controls** on mobile
- **An autopilot** you can sit back and watch

## How the autopilot thinks

The AI pilot is a mix of a few classic techniques working together:

- **A\* search** finds a safe route through the asteroid field.
- **Model predictive control** plans a few moves ahead and keeps adjusting as things change.
- **Collision prediction** works out which asteroids will actually hit the ship, and when.
- **Predictive aiming** shoots where an asteroid is going to be, not where it is now.
- **Threat assessment** ranks nearby dangers so the ship deals with the worst one first.
- **Adaptive risk management** makes the ship more cautious or more aggressive depending on how bad things look.

## Built with

Python, Pygame-ce, NumPy, Pygbag (for the WebAssembly build), and Vercel for hosting.

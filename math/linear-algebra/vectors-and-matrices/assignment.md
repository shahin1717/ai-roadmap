# Assignment: A Tiny Movie Recommender

Use dot products and matrix-vector multiplication to recommend movies. It takes about 30-45 minutes and needs only NumPy. Back to the [topic](README.md).

## The setup

Each movie is described by a vector of four scores from 0 to 5: **action, romance, comedy, sci-fi**. These numbers are invented for this exercise.

| Movie | Action | Romance | Comedy | Sci-fi |
|---|---|---|---|---|
| Starfall | 5 | 0 | 1 | 5 |
| Love in Lisbon | 0 | 5 | 2 | 0 |
| Laugh Track | 1 | 2 | 5 | 0 |
| Iron Horizon | 5 | 1 | 0 | 3 |
| Moonlit Letters | 0 | 4 | 1 | 1 |
| Robo Roommates | 2 | 0 | 4 | 4 |

A user's taste is also a vector in the same four dimensions. Ana's taste is `[2, 1, 2, 2]`: she enjoys action, comedy, and sci-fi about equally, with a little romance.

## Tasks

1. **Build the data.** Put the six movies into a NumPy array `M` of shape $6 \times 4$ (one row per movie) and Ana's taste into a vector `ana` of length 4.
2. **Score with one multiplication.** Compute `scores = M @ ana`. Explain in one sentence why each entry of `scores` is a dot product, and say which movie ranks first.
3. **Ignore the "loudness".** Dot-product scores favour movies with big numbers everywhere. Compute the *cosine similarity* between Ana and each movie (divide each dot product by the two lengths) and rank the movies again. Does the top pick change? Which movie moved, and why do you think so? (Hint: look at how large the numbers in each movie's row are.)
4. **A second user.** Create a taste vector for someone with different preferences and rank the movies for them. Check that the recommendations change in a sensible way.
5. **Mix the features.** The matrix
   ```
   F = [[1, 0, 1, 1],     # "excitement" = action + comedy + sci-fi
        [0, 1, 1, 0]]     # "warmth"     = romance + comedy
   ```
   is $2 \times 4$. Compute `M @ F.T` to turn every movie into just two numbers, "excitement" and "warmth". (`F.T` flips the matrix so the shapes fit; the matrix operations topic explains it properly.) Then draw a scatter plot with one labelled point per movie. Which movies sit close together?

## What to hand in

A short notebook or script that runs top to bottom, plus 3-5 sentences answering the questions in tasks 2, 3, and 5.

## Stretch (optional)

Add a seventh movie of your own and predict where it will land on the scatter plot before you compute it.

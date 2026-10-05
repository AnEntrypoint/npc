import { BLOB } from './blob.js';
import { REALM } from './realm.js';

export const GAMES = { [BLOB.id]: BLOB, [REALM.id]: REALM };

export function registerGame(game) {
  GAMES[game.id] = game;
  return game;
}

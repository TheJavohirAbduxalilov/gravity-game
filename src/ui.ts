import type { BodySnapshot } from "./gpu-engine";

function formatNumber(value: number, digits = 1): string {
  const safeValue = Math.abs(value) < 0.05 ? 0 : value;
  return safeValue.toFixed(digits);
}

function getBodyColor(mass: number, vx: number, vy: number): string {
  const speed = Math.hypot(vx, vy);
  const t = 1.0 - Math.exp(-(mass * 0.012 + speed * 0.005));
  let r = 0, g = 0, b = 0;
  if (t < 0.35) {
    const k = t / 0.35;
    r = 100 + (180 - 100) * k;
    g = 105 + (30 - 105) * k;
    b = 115 + (20 - 115) * k;
  } else if (t < 0.7) {
    const k = (t - 0.35) / 0.35;
    r = 180 + (255 - 180) * k;
    g = 30 + (115 - 30) * k;
    b = 20 + (30 - 20) * k;
  } else {
    const k = (t - 0.7) / 0.3;
    r = 255;
    g = 115 + (215 - 115) * k;
    b = 30 + (100 - 30) * k;
  }
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

const STAR_ICON = `<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path d="M8 1.7l1.85 3.9 4.25.5-3.15 2.95.83 4.25L8 11.65 4.22 13.3l.83-4.25L1.9 6.1l4.25-.5z" fill="currentColor" stroke="currentColor" stroke-width=".8" stroke-linejoin="round"/></svg>`;

type CardRefs = {
  card: HTMLElement;
  nameText: HTMLElement;
  mass: HTMLElement;
  speed: HTMLElement;
  favorite: HTMLButtonElement;
  color: HTMLElement;
};

export class BodiesSidebar {
  private readonly container: HTMLElement;
  private readonly empty: HTMLElement;
  private readonly cards = new Map<number, CardRefs>();
  private readonly favorites = new Set<number>();
  private selectedId: number | null = null;
  private inspectedBodyId: number | null = null;
  private scrollToSelected = false;
  private readonly onFocus: (id: number) => void;
  private readonly onDelete: (id: number) => void;
  private readonly onRename: (id: number, name: string) => void;
  private lastSnapshots: BodySnapshot[] = [];
  private searchQuery = "";

  constructor(
    onFocus: (id: number) => void,
    onDelete: (id: number) => void,
    onRename: (id: number, name: string) => void
  ) {
    this.container = document.querySelector<HTMLElement>("#bodies-list")!;
    this.empty = document.querySelector<HTMLElement>("#empty-bodies")!;
    this.onFocus = onFocus;
    this.onDelete = onDelete;
    this.onRename = onRename;

    const searchInput = document.querySelector<HTMLInputElement>("#body-search");
    if (searchInput) {
      searchInput.addEventListener("input", () => {
        this.searchQuery = searchInput.value;
        this.update(this.lastSnapshots);
      });
    }
  }

  update(bodies: BodySnapshot[]): void {
    this.lastSnapshots = bodies;
    const query = this.searchQuery.toLowerCase().trim();
    const filteredBodies = query
      ? bodies.filter((body) => body.name.toLowerCase().includes(query))
      : bodies;

    const liveIds = new Set(filteredBodies.map((body) => body.id));
    const allLiveIds = new Set(bodies.map((body) => body.id));

    // Удаляем карточки исчезнувших тел (слияние/удаление или отфильтрованных).
    for (const [id, refs] of this.cards) {
      if (!liveIds.has(id)) {
        refs.card.remove();
        this.cards.delete(id);
        if (!allLiveIds.has(id)) {
          this.favorites.delete(id);
        }
        if (this.selectedId === id && !allLiveIds.has(id)) {
          this.selectedId = null;
        }
      }
    }

    // Создаём недостающие карточки и обновляем только текстовые значения у
    // существующих — без полной перерисовки DOM (список не дёргается).
    for (const body of filteredBodies) {
      let refs = this.cards.get(body.id);
      if (!refs) {
        refs = this.createCard(body);
        this.cards.set(body.id, refs);
        this.container.appendChild(refs.card);
      }
      this.updateValues(refs, body);
    }

    this.reorder();
    this.applySelection();
    this.empty.hidden = filteredBodies.length > 0;
    this.updateInspector(bodies);
  }

  setSelected(id: number | null): void {
    if (this.selectedId === id) return;
    this.selectedId = id;
    this.scrollToSelected = id !== null;
    this.applySelection();
    this.updateInspector(this.lastSnapshots);
  }

  private updateInspector(bodies: BodySnapshot[]): void {
    const inspectContent = document.querySelector<HTMLElement>("#inspect-content");
    if (!inspectContent) return;

    if (this.selectedId === null) {
      inspectContent.innerHTML = `<div class="inspect-placeholder">Выберите тело на карте или в списке слева для просмотра свойств</div>`;
      this.inspectedBodyId = null;
      return;
    }

    const body = bodies.find((b) => b.id === this.selectedId);
    if (!body) {
      inspectContent.innerHTML = `<div class="inspect-placeholder">Тело не найдено или уничтожено</div>`;
      this.inspectedBodyId = null;
      return;
    }

    const speed = Math.hypot(body.velocity.x, body.velocity.y);
    const color = getBodyColor(body.mass, body.velocity.x, body.velocity.y);

    if (this.inspectedBodyId !== body.id) {
      this.inspectedBodyId = body.id;
      inspectContent.innerHTML = `
        <div class="inspect-title-wrap">
          <span class="body-color" style="background: ${color}; color: ${color};"></span>
          <input type="text" class="inspect-name-input" placeholder="Название" maxlength="24" aria-label="Имя тела">
          <button class="inspect-favorite-toggle" type="button" title="В избранное">${STAR_ICON}</button>
        </div>
        <div class="inspect-property-table">
          <div class="inspect-row">
            <span class="inspect-label">Масса</span>
            <strong class="inspect-val inspect-val-mass"></strong>
          </div>
          <div class="inspect-row">
            <span class="inspect-label">Скорость</span>
            <strong class="inspect-val inspect-val-speed"></strong>
          </div>
          <div class="inspect-row">
            <span class="inspect-label">Координата X</span>
            <strong class="inspect-val inspect-val-x"></strong>
          </div>
          <div class="inspect-row">
            <span class="inspect-label">Координата Y</span>
            <strong class="inspect-val inspect-val-y"></strong>
          </div>
        </div>
        <div class="inspect-actions">
          <button class="inspect-action-btn btn-focus" type="button" title="Сфокусировать камеру">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M12 8c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm-7 4c0-3.87 3.13-7 7-7s7 3.13 7 7-3.13 7-7 7-7-3.13-7-7zm7-9C6.48 3 2 7.48 2 13s4.48 10 10 10 10-4.48 10-10S17.52 3 12 3z" fill="currentColor"/></svg>
            Фокус
          </button>
          <button class="inspect-action-btn btn-delete" type="button" title="Удалить тело">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" fill="currentColor"/></svg>
            Удалить
          </button>
        </div>`;

      const nameInput = inspectContent.querySelector<HTMLInputElement>(".inspect-name-input")!;
      nameInput.value = body.name;
      nameInput.addEventListener("input", () => {
        const newName = nameInput.value.trim() || `Небесное тело ${body.id + 1}`;
        this.onRename(body.id, newName);
      });

      const favBtn = inspectContent.querySelector<HTMLButtonElement>(".inspect-favorite-toggle")!;
      favBtn.addEventListener("click", () => {
        this.toggleFavorite(body.id);
      });

      inspectContent.querySelector(".btn-focus")?.addEventListener("click", () => this.onFocus(body.id));
      inspectContent.querySelector(".btn-delete")?.addEventListener("click", () => this.onDelete(body.id));
    }

    const colorDot = inspectContent.querySelector<HTMLElement>(".body-color")!;
    const massVal = inspectContent.querySelector<HTMLElement>(".inspect-val-mass")!;
    const speedVal = inspectContent.querySelector<HTMLElement>(".inspect-val-speed")!;
    const xVal = inspectContent.querySelector<HTMLElement>(".inspect-val-x")!;
    const yVal = inspectContent.querySelector<HTMLElement>(".inspect-val-y")!;
    const favBtn = inspectContent.querySelector<HTMLButtonElement>(".inspect-favorite-toggle")!;
    const nameInput = inspectContent.querySelector<HTMLInputElement>(".inspect-name-input")!;

    if (colorDot) {
      colorDot.style.background = color;
      colorDot.style.color = color;
    }
    if (massVal) massVal.textContent = formatNumber(body.mass, 1);
    if (speedVal) speedVal.textContent = formatNumber(speed, 1);
    if (xVal) xVal.textContent = formatNumber(body.position.x, 0);
    if (yVal) yVal.textContent = formatNumber(body.position.y, 0);

    const isFav = this.favorites.has(body.id);
    favBtn.classList.toggle("is-active", isFav);
    favBtn.setAttribute("aria-pressed", String(isFav));

    if (document.activeElement !== nameInput) {
      nameInput.value = body.name;
    }
  }

  // Подсветка активной карточки + автоскролл к ней.
  private applySelection(): void {
    for (const [cardId, refs] of this.cards) {
      refs.card.classList.toggle("is-selected", cardId === this.selectedId);
    }
    if (this.scrollToSelected && this.selectedId !== null) {
      const refs = this.cards.get(this.selectedId);
      if (refs) {
        refs.card.scrollIntoView({ block: "nearest", behavior: "smooth" });
        this.scrollToSelected = false;
      }
    }
  }

  private rank(id: number): number {
    return this.favorites.has(id) ? 0 : 1;
  }

  // Избранные закреплены вверху, остальные — по id. Переставляем уже
  // существующие DOM-узлы, не создавая их заново.
  private reorder(): void {
    const ordered = [...this.cards.entries()].sort(([idA], [idB]) => {
      const rankA = this.rank(idA);
      const rankB = this.rank(idB);
      return rankA !== rankB ? rankA - rankB : idA - idB;
    });
    ordered.forEach(([, refs], index) => {
      if (this.container.children[index] !== refs.card) {
        this.container.insertBefore(refs.card, this.container.children[index] ?? null);
      }
    });
  }

  private updateValues(refs: CardRefs, body: BodySnapshot): void {
    refs.nameText.textContent = body.name;
    refs.mass.textContent = `${formatNumber(body.mass, 1)} M`;
    refs.speed.textContent = formatNumber(Math.hypot(body.velocity.x, body.velocity.y), 1);
    const c = getBodyColor(body.mass, body.velocity.x, body.velocity.y);
    refs.color.style.background = c;
    refs.color.style.color = c;
  }

  private toggleFavorite(id: number): void {
    if (this.favorites.has(id)) this.favorites.delete(id);
    else this.favorites.add(id);
    const refs = this.cards.get(id);
    if (refs) {
      const active = this.favorites.has(id);
      refs.favorite.classList.toggle("is-active", active);
      refs.favorite.setAttribute("aria-pressed", String(active));
      refs.card.classList.toggle("is-favorite", active);
    }
    this.reorder();

    const inspectContent = document.querySelector<HTMLElement>("#inspect-content");
    if (inspectContent && this.selectedId === id) {
      const favBtn = inspectContent.querySelector<HTMLButtonElement>(".inspect-favorite-toggle");
      if (favBtn) {
        const isFav = this.favorites.has(id);
        favBtn.classList.toggle("is-active", isFav);
        favBtn.setAttribute("aria-pressed", String(isFav));
      }
    }
  }

  private createCard(body: BodySnapshot): CardRefs {
    const card = document.createElement("article");
    card.className = "body-card";
    card.innerHTML = `
      <span class="body-color"></span>
      <span class="body-name"></span>
      <span class="body-mass"></span>
      <span class="body-speed"></span>
      <button class="favorite-toggle" type="button" aria-pressed="false" aria-label="В избранное" title="В избранное">${STAR_ICON}</button>
    `;

    const refs: CardRefs = {
      card,
      nameText: card.querySelector<HTMLElement>(".body-name")!,
      mass: card.querySelector<HTMLElement>(".body-mass")!,
      speed: card.querySelector<HTMLElement>(".body-speed")!,
      favorite: card.querySelector<HTMLButtonElement>(".favorite-toggle")!,
      color: card.querySelector<HTMLElement>(".body-color")!,
    };

    card.addEventListener("click", () => {
      this.onFocus(body.id);
    });

    refs.favorite.addEventListener("click", (event) => {
      event.stopPropagation();
      this.toggleFavorite(body.id);
    });

    if (this.selectedId === body.id) card.classList.add("is-selected");
    const isFav = this.favorites.has(body.id);
    refs.favorite.classList.toggle("is-active", isFav);
    refs.favorite.setAttribute("aria-pressed", String(isFav));
    card.classList.toggle("is-favorite", isFav);

    return refs;
  }
}

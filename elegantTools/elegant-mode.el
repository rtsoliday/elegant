;;; elegant-mode.el --- Major mode for ELEGANT input files -*- lexical-binding: t; -*-

(require 'cl-lib)

(defgroup elegant nil
  "Editing support for ELEGANT input files."
  :group 'languages)

(defcustom elegant-indent-offset 4
  "Indentation width for qualifier lines."
  :type 'integer)

;; ----------------------------
;; Syntax & Highlighting
;; ----------------------------

(defvar elegant-mode-syntax-table
  (let ((st (make-syntax-table)))
    (modify-syntax-entry ?! "<" st)
    (modify-syntax-entry ?\n ">" st)
    st))

(defvar elegant-font-lock-keywords
  `(
    ("^\\s-*\\(&\\)\\([A-Za-z_][A-Za-z0-9_]*\\)"
     (1 font-lock-builtin-face)
     (2 font-lock-function-name-face))
    ("^\\s-*\\(&end\\)\\(?:\\s-+\\|$\\)"
     (1 font-lock-builtin-face))
    ("^\\s-*\\([A-Za-z_][A-Za-z0-9_]*\\)\\s-*="
     (1 font-lock-variable-name-face))
    ))

;; ----------------------------
;; Block Detection
;; ----------------------------

(defun elegant--block-region ()
  (save-excursion
    (let ((pos (point)))
      (when (re-search-backward "^\\s-*&\\([A-Za-z_][A-Za-z0-9_]*\\)" nil t)
        (unless (looking-at "^\\s-*&end\\(?:\\s-+\\|$\\)")
          (let ((start (match-beginning 0)))
            (goto-char start)
            (when (re-search-forward "^\\s-*&end\\(?:\\s-+\\|$\\)" nil t)
              (let ((end (line-end-position)))
                (when (and (<= start pos) (<= pos end))
                  (cons start end))))))))))

(defun elegant--current-namelist ()
  (save-excursion
    (let ((blk (elegant--block-region)))
      (when blk
        (goto-char (car blk))
        (when (looking-at "^\\s-*&\\([A-Za-z_][A-Za-z0-9_]*\\)")
          (match-string-no-properties 1))))))

(defun elegant--qualifiers-for (namelist)
  (when (and namelist (boundp 'elegant-namelist-qualifiers))
    (cdr (assoc namelist elegant-namelist-qualifiers))))

(defun elegant--bounds-of-symbol ()
  (let ((pt (point)))
    (save-excursion
      (let ((end (progn (skip-chars-forward "A-Za-z0-9_") (point)))
            (beg (progn (goto-char pt)
                        (skip-chars-backward "A-Za-z0-9_")
                        (point))))
        (cons beg end)))))

;; ----------------------------
;; Indentation
;; ----------------------------

(defun elegant-indent-line ()
  "Indent current line appropriately for ELEGANT."
  (interactive)
  (let ((indent 0)
        (pos (- (point-max) (point))))
    (cond
     ;; &header lines
     ((save-excursion
        (beginning-of-line)
        (looking-at "^\\s-*&\\([A-Za-z_][A-Za-z0-9_]*\\)"))
      (setq indent 0))
     ;; &end lines
     ((save-excursion
        (beginning-of-line)
        (looking-at "^\\s-*&end\\(?:\\s-+\\|$\\)"))
      (setq indent 0))
     ;; inside block
     ((elegant--current-namelist)
      (setq indent elegant-indent-offset))
     (t
      (setq indent 0)))
    (indent-line-to indent)
    ;; preserve cursor position
    (when (> (- (point-max) pos) (point))
      (goto-char (- (point-max) pos)))))

;; ----------------------------
;; Completion
;; ----------------------------

(defun elegant-completion-at-point ()
  (let* ((bnds (elegant--bounds-of-symbol))
         (beg (car bnds))
         (end (cdr bnds)))
    (cond
     ;; Complete namelist names
     ((save-excursion
        (beginning-of-line)
        (looking-at "^\\s-*&"))
      (save-excursion
        (beginning-of-line)
        (re-search-forward "^\\s-*&" (line-end-position) t)
        (setq beg (point))
        (skip-chars-forward "A-Za-z0-9_")
        (setq end (point)))
      (when (boundp 'elegant-namelist-commands)
        (list beg end elegant-namelist-commands
              :exclusive 'no)))

     ;; Complete qualifiers
     (t
      (let ((nl (elegant--current-namelist)))
        (when (and nl
                   (not (save-excursion
                          (beginning-of-line)
                          (looking-at "^\\s-*&"))))
          (let ((qs (elegant--qualifiers-for nl)))
            (when (and qs (listp qs))
              (list beg end qs :exclusive 'no)))))))))

;; ----------------------------
;; Auto-insert &end
;; ----------------------------

(defun elegant--namelist-has-end-ahead-p ()
  (save-excursion
    (forward-line 1)
    (catch 'done
      (while (not (eobp))
        (cond
         ((looking-at "^\\s-*&end\\(?:\\s-+\\|$\\)") (throw 'done t))
         ((looking-at "^\\s-*&\\([A-Za-z_][A-Za-z0-9_]*\\)") (throw 'done nil)))
        (forward-line 1))
      nil)))

(defun elegant-newline-and-maybe-insert-end ()
  (interactive)
  (let ((on-header
         (save-excursion
           (beginning-of-line)
           (and (looking-at "^\\s-*&\\([A-Za-z_][A-Za-z0-9_]*\\)")
                (not (looking-at "^\\s-*&end\\(?:\\s-+\\|$\\)"))))))
    (newline)
    (when (and on-header (not (elegant--namelist-has-end-ahead-p)))
      (save-excursion
        (newline)
        (insert "&end\n\n")))
    (elegant-indent-line)))

;; ----------------------------
;; Keymap
;; ----------------------------

(defvar elegant-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "TAB") #'completion-at-point)
    (define-key map (kbd "<tab>") #'completion-at-point)
    (define-key map (kbd "RET") #'elegant-newline-and-maybe-insert-end)
    map))

;; ----------------------------
;; Mode Definition
;; ----------------------------

;;;###autoload
(define-derived-mode elegant-mode fundamental-mode "ELEGANT"
  "Major mode for editing ELEGANT input files."
  :syntax-table elegant-mode-syntax-table
  (setq-local font-lock-defaults '(elegant-font-lock-keywords))
  (setq-local indent-line-function #'elegant-indent-line)
  (setq-local comment-start "!")
  (setq-local comment-end "")
  (add-hook 'completion-at-point-functions
            #'elegant-completion-at-point nil t))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.ele\\'" . elegant-mode))

(provide 'elegant-mode)
;;; elegant-mode.el ends here

;;; elegant-lattice-mode.el --- Major mode for ELEGANT lattice files (*.lte, *.lat) -*- lexical-binding: t; -*-

(require 'cl-lib)
(require 'thingatpt)

(defgroup elegant-lattice nil
  "Editing ELEGANT lattice files."
  :group 'languages)

(defcustom elegant-lattice-indent-offset 2
  "Indentation offset for continued lines."
  :type 'integer
  :group 'elegant-lattice)

(defvar elegant-lattice-mode-syntax-table
  (let ((st (make-syntax-table)))
    (modify-syntax-entry ?! "<" st)
    (modify-syntax-entry ?\n ">" st)
    (modify-syntax-entry ?_ "w" st)
    st))

(defvar elegant-lattice--type-regexp nil
  "Cached regexp for element types.")

(defvar elegant-lattice-font-lock-keywords nil
  "Font-lock keywords for `elegant-lattice-mode`.")

(defun elegant-lattice--ensure-data ()
  (unless (featurep 'elegant-lattice-data)
    (unless (require 'elegant-lattice-data nil t)
      (user-error "Cannot load elegant-lattice-data.el. Ensure it is in `load-path`")))
  (unless elegant-lattice--type-regexp
    (setq elegant-lattice--type-regexp
          (regexp-opt elegant-lattice-element-types 'words)))
  (unless elegant-lattice-font-lock-keywords
    (setq elegant-lattice-font-lock-keywords
          `((,elegant-lattice--type-regexp . font-lock-type-face)
            ("^[ \t]*\\([A-Za-z][A-Za-z0-9_]*\\)[ \t]*:" (1 font-lock-function-name-face))
            ("\\_<[A-Za-z][A-Za-z0-9_]*\\_>[ \t]*=" (0 font-lock-variable-name-face))
            ("\\_<[-+]?[0-9]*\\.?[0-9]+\\([eEdD][-+]?[0-9]+\\)?\\_>" . font-lock-constant-face)))))

(defun elegant-lattice--line-to-point ()
  (buffer-substring-no-properties (line-beginning-position) (point)))

(defun elegant-lattice--bounds ()
  (let ((end (point)))
    (save-excursion
      (skip-syntax-backward "w_")
      (cons (point) end))))

(defun elegant-lattice--element-type-on-line ()
  "Return element TYPE on current line as uppercase, or nil."
  (save-excursion
    (beginning-of-line)
    (when (re-search-forward
           "^[ \t]*[A-Za-z][A-Za-z0-9_]*[ \t]*:[ \t]*\\([A-Za-z][A-Za-z0-9_]*\\)"
           (line-end-position) t)
      (upcase (match-string-no-properties 1)))))

(defun elegant-lattice--type-context-p ()
  (let ((s (elegant-lattice--line-to-point)))
    (and (string-match "^[ \t]*[A-Za-z][A-Za-z0-9_]*[ \t]*:" s)
         (not (string-match "," s)))))

(defun elegant-lattice--param-context-p ()
  (let ((s (elegant-lattice--line-to-point)))
    (and (string-match "^[ \t]*[A-Za-z][A-Za-z0-9_]*[ \t]*:" s)
         (string-match "," s)
         (not (elegant-lattice--after-equals-p)))))

(defun elegant-lattice--apply-case-style (typed candidate)
  (cond
   ((string= typed (downcase typed)) (downcase candidate))
   ((string= typed (capitalize typed)) (capitalize (downcase candidate)))
   (t candidate)))

(defun elegant-lattice--collect-types (prefix)
  (cl-remove-if-not (lambda (x) (string-prefix-p prefix x t))
                    elegant-lattice-element-types))

(defun elegant-lattice--params-for-type (etype)
  (cdr (assoc-string etype elegant-lattice-element-params t)))

(defun elegant-lattice--collect-params (etype prefix)
  (cl-remove-if-not (lambda (x) (string-prefix-p prefix x t))
                    (or (elegant-lattice--params-for-type etype) '())))

(defun elegant-lattice--pretty-unit (unit)
  "Convert SDDS-style unit markup to readable ASCII superscripts/subscripts."
  (let ((u (or unit "")))
    (setq u (replace-regexp-in-string "\\$a\\(-?[0-9]+\\)\\$n" "^\\1" u))
    (setq u (replace-regexp-in-string "\\$b\\(-?[0-9]+\\)\\$n" "_\\1" u))
    u))

(defun elegant-lattice--normalize-param-doc (ud)
  "Normalize UD into (UNIT TYPE DOC)."
  (cond
   ((and (consp ud) (stringp (car ud)) (stringp (cdr ud)))
    (list (car ud) "" (cdr ud)))
   ((and (consp ud) (stringp (car ud)) (consp (cdr ud))
         (stringp (car (cdr ud))) (stringp (cdr (cdr ud))))
    (list (car ud) (car (cdr ud)) (cdr (cdr ud))))
   ((and (listp ud) (>= (length ud) 3))
    (list (nth 0 ud) (nth 1 ud) (nth 2 ud)))
   (t (list "" "" ""))))

(defun elegant-lattice--annotate-type (cand)
  (let* ((canon (upcase cand))
         (doc (cdr (assoc-string canon elegant-lattice-element-docs t))))
    (when (and doc (> (length doc) 0))
      (concat "  " doc))))

(defun elegant-lattice--annotate-param (etype cand)
  (let* ((key (cons (upcase etype) (upcase cand)))
         (ud (cdr (assoc key elegant-lattice-param-docs))))
    (when ud
      (pcase (elegant-lattice--normalize-param-doc ud)
        (`(,unit ,ptype ,pdoc)
         (setq unit (elegant-lattice--pretty-unit unit))
         (concat "  "
                 (if (and unit (> (length unit) 0)) (format "[%s] " unit) "")
                 (if (and ptype (> (length ptype) 0)) (format "(%s) " ptype) "")
                 (or pdoc "")))))))

(defun elegant-lattice--capf ()
  (elegant-lattice--ensure-data)
  (let* ((bounds (elegant-lattice--bounds))
         (beg (car bounds))
         (end (cdr bounds)))
    (cond
     ((elegant-lattice--type-context-p)
      (list beg end
            (completion-table-dynamic
             (lambda (prefix)
               (mapcar (lambda (c) (elegant-lattice--apply-case-style prefix c))
                       (elegant-lattice--collect-types prefix))))
            :annotation-function #'elegant-lattice--annotate-type
            :exclusive 'no))
     ((elegant-lattice--param-context-p)
      (let ((etype (elegant-lattice--element-type-on-line)))
        (when etype
          (list beg end
                (completion-table-dynamic
                 (lambda (prefix)
                   (mapcar (lambda (c) (elegant-lattice--apply-case-style prefix c))
                           (elegant-lattice--collect-params etype prefix))))
                :annotation-function (lambda (cand) (elegant-lattice--annotate-param etype cand))
                :exclusive 'no))))
     (t nil))))

(defun elegant-lattice--param-before-equals ()
  (save-excursion
    (let ((etype (elegant-lattice--element-type-on-line)))
      (when etype
        (skip-chars-backward " \t")
        (when (and (> (point) (line-beginning-position))
                   (eq (char-before) ?=))
          (backward-char 1)
          (skip-chars-backward " \t")
          (let ((end (point)))
            (skip-syntax-backward "w_")
            (let ((p (buffer-substring-no-properties (point) end)))
              (when (and p (> (length p) 0))
                (cons (upcase etype) (upcase p))))))))))

(defun elegant-lattice--after-equals-p ()
  (save-excursion
    (skip-chars-backward " \t")
    (and (> (point) (line-beginning-position))
         (eq (char-before) ?=))))

(defun elegant-lattice-show-param-info ()
  (interactive)
  (elegant-lattice--ensure-data)
  (let ((pair (elegant-lattice--param-before-equals)))
    (if (not pair)
        (message "No PARAM= at point")
      (let* ((etype (car pair))
             (pname (cdr pair))
             (ud (cdr (assoc (cons etype pname) elegant-lattice-param-docs))))
        (if (not ud)
            (message "%s.%s: (no metadata)" etype pname)
          (pcase (elegant-lattice--normalize-param-doc ud)
            (`(,unit ,ptype ,pdoc)
             (setq unit (elegant-lattice--pretty-unit unit))
             (message "%s.%s%s%s%s"
                      etype pname
                      (if (and unit (> (length unit) 0)) (format "  units=%s" unit) "")
                      (if (and ptype (> (length ptype) 0)) (format "  type=%s" ptype) "")
                      (if (and pdoc (> (length pdoc) 0)) (format "  %s" pdoc) "")))))))))

(defun elegant-lattice-tab ()
  (interactive)
  (cond
   ((elegant-lattice--after-equals-p)
    (elegant-lattice-show-param-info))
   ((and (boundp 'completion-at-point-functions)
         (run-hook-with-args-until-success 'completion-at-point-functions))
    (completion-at-point))
   (t
    (indent-for-tab-command))))

(defun elegant-lattice-indent-line ()
  (interactive)
  (let ((indent 0))
    (indent-line-to indent)))

(defvar elegant-lattice-mode-map
  (let ((map (make-sparse-keymap)))
    (define-key map (kbd "TAB") #'elegant-lattice-tab)
    (define-key map (kbd "<tab>") #'elegant-lattice-tab)
    map))

;;;###autoload
(define-derived-mode elegant-lattice-mode prog-mode "ELEGANT-LTE"
  "Major mode for editing ELEGANT lattice files."
  :syntax-table elegant-lattice-mode-syntax-table
  (setq-local completion-ignore-case t)
  (setq-local case-fold-search t)
  (elegant-lattice--ensure-data)
  (setq-local font-lock-defaults '(elegant-lattice-font-lock-keywords))
  (setq-local comment-start "!")
  (setq-local comment-end "")
  (setq-local indent-line-function #'elegant-lattice-indent-line)
  (add-hook 'completion-at-point-functions #'elegant-lattice--capf nil t))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.lte\\'" . elegant-lattice-mode))
;;;###autoload
(add-to-list 'auto-mode-alist '("\\.lat\\'" . elegant-lattice-mode))

(provide 'elegant-lattice-mode)
;;; elegant-lattice-mode.el ends here
